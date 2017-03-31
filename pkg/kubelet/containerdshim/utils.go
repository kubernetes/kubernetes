package containerdshim

import (
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	gocontext "context"

	"github.com/docker/containerd/api/services/shim"
	"github.com/docker/containerd/api/types/container"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/tonistiigi/fifo"
	"google.golang.org/grpc"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
)

const (
	// kubePrefix is used to identify the containers/sandboxes on the node managed by kubelet
	kubePrefix = "k8s"
	// sandboxContainerName is a string to include in the docker container so
	// that users can easily identify the sandboxes.
	sandboxContainerName = leaky.PodInfraContainerName
	// Delimiter used to construct docker container names.
	nameDelimiter = "_"
)

var rwm = "rwm"

func defaultOCISpec(id string, args []string, rootfs string, tty bool) *specs.Spec {
	return &specs.Spec{
		Version: specs.Version,
		Platform: specs.Platform{
			OS:   runtime.GOOS,
			Arch: runtime.GOARCH,
		},
		Root: specs.Root{
			Path:     rootfs,
			Readonly: true,
		},
		Process: specs.Process{
			Args: args,
			Env: []string{
				"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			},
			Terminal:        tty,
			Cwd:             "/",
			NoNewPrivileges: true,
		},
		Mounts: []specs.Mount{
			{
				Destination: "/proc",
				Type:        "proc",
				Source:      "proc",
			},
			{
				Destination: "/dev",
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
			},
			{
				Destination: "/dev/pts",
				Type:        "devpts",
				Source:      "devpts",
				Options:     []string{"nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620", "gid=5"},
			},
			{
				Destination: "/dev/shm",
				Type:        "tmpfs",
				Source:      "shm",
				Options:     []string{"nosuid", "noexec", "nodev", "mode=1777", "size=65536k"},
			},
			{
				Destination: "/dev/mqueue",
				Type:        "mqueue",
				Source:      "mqueue",
				Options:     []string{"nosuid", "noexec", "nodev"},
			},
			{
				Destination: "/sys",
				Type:        "sysfs",
				Source:      "sysfs",
				Options:     []string{"nosuid", "noexec", "nodev"},
			},
			{
				Destination: "/run",
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
			},
			{
				Destination: "/etc/resolv.conf",
				Type:        "bind",
				Source:      "/etc/resolv.conf",
				Options:     []string{"rbind", "ro"},
			},
			{
				Destination: "/etc/hosts",
				Type:        "bind",
				Source:      "/etc/hosts",
				Options:     []string{"rbind", "ro"},
			},
			{
				Destination: "/etc/localtime",
				Type:        "bind",
				Source:      "/etc/localtime",
				Options:     []string{"rbind", "ro"},
			},
		},
		Hostname: id,
		Linux: &specs.Linux{
			Resources: &specs.LinuxResources{
				Devices: []specs.LinuxDeviceCgroup{
					{
						Allow:  false,
						Access: &rwm,
					},
				},
			},
			Namespaces: []specs.LinuxNamespace{
				{
					Type: "pid",
				},
				{
					Type: "ipc",
				},
				{
					Type: "uts",
				},
				{
					Type: "mount",
				},
				{
					Type: "network",
				},
			},
		},
	}
}

func addOrReplaceNamespace(namespaces []specs.LinuxNamespace, t specs.LinuxNamespaceType, path string) []specs.LinuxNamespace {
	var namespace *specs.LinuxNamespace
	for i := range namespaces {
		if namespaces[i].Type == t {
			namespace = &namespaces[i]
			break
		}
	}
	if namespace == nil {
		namespaces = append(namespaces, specs.LinuxNamespace{Type: t, Path: path})
	} else {
		namespace.Path = path
	}
	return namespaces
}

func ensureContainerDir(id string) (string, error) {
	dir := getContainerDir(id)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return "", err
	}
	return dir, nil
}

func getContainerDir(id string) string {
	return filepath.Join(containerdCRIRoot, id)
}

type stream struct {
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser
}

func (s *stream) Close() {
	s.stdin.Close()
	s.stdout.Close()
	s.stderr.Close()
}

func prepareStdio(stdin, stdout, stderr string, console bool) (*stream, error) {
	ctx := gocontext.Background()
	var s stream
	var err error
	s.stdin, err = fifo.OpenFifo(ctx, stdin, syscall.O_WRONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700)
	if err != nil {
		return nil, err
	}
	defer func(c io.Closer) {
		if err != nil {
			c.Close()
		}
	}(s.stdin)

	s.stdout, err = fifo.OpenFifo(ctx, stdout, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700)
	if err != nil {
		return nil, err
	}
	defer func(c io.Closer) {
		if err != nil {
			c.Close()
		}
	}(s.stdout)

	s.stderr, err = fifo.OpenFifo(ctx, stderr, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700)
	if err != nil {
		return nil, err
	}
	defer func(c io.Closer) {
		if err != nil {
			c.Close()
		}
	}(s.stderr)

	return &s, nil
}

func getShimClient(id string) (shim.ShimClient, error) {
	bindSocket := filepath.Join(containerdVarRun, "linux", id, shimbindSocket)
	dialOpts := []grpc.DialOption{
		grpc.WithInsecure(),
		grpc.WithTimeout(100 * time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", bindSocket, timeout)
		}),
	}
	conn, err := grpc.Dial(fmt.Sprintf("unix://%s", bindSocket), dialOpts...)
	if err != nil {
		return nil, err
	}
	return shim.NewShimClient(conn), nil
}

func statusToContainer(s *runtimeapi.ContainerStatus) *runtimeapi.Container {
	return &runtimeapi.Container{
		Id:          s.Id,
		Metadata:    s.Metadata,
		Image:       s.Image,
		ImageRef:    s.ImageRef,
		State:       s.State,
		CreatedAt:   s.CreatedAt,
		Labels:      s.Labels,
		Annotations: s.Annotations,
	}
}

func statusToSandbox(s *runtimeapi.PodSandboxStatus) *runtimeapi.PodSandbox {
	return &runtimeapi.PodSandbox{
		Id:          s.Id,
		Metadata:    s.Metadata,
		State:       s.State,
		CreatedAt:   s.CreatedAt,
		Labels:      s.Labels,
		Annotations: s.Annotations,
	}
}

func toCRIContainer(c *container.Container) (*runtimeapi.Container, error) {
	metadata, err := dockershim.ParseContainerName(c.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to parse container id %s: %v", c.ID, err)
	}
	return &runtimeapi.Container{
		Id:       c.ID,
		Metadata: metadata,
		State:    toCRIContainerState(c.Status),
		// TODO: Add image information.
		// TODO: Provide correct creation time.
		CreatedAt: time.Now().Unix(),
		// TODO: Add label and annotation.
		// TODO: Add image in either local cache or wait for metadata store.
	}, nil
}

func toCRIContainerState(status container.Status) runtimeapi.ContainerState {
	switch status {
	case container.Status_CREATED:
		return runtimeapi.ContainerState_CONTAINER_CREATED
	case container.Status_RUNNING:
		return runtimeapi.ContainerState_CONTAINER_RUNNING
	case container.Status_STOPPED:
		return runtimeapi.ContainerState_CONTAINER_EXITED
	default:
		return runtimeapi.ContainerState_CONTAINER_UNKNOWN
	}
}
