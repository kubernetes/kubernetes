// +build !windows

package client

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/linux/shim"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/reaper"
	"github.com/containerd/containerd/sys"
	google_protobuf "github.com/golang/protobuf/ptypes/empty"
	"google.golang.org/grpc"
)

var empty = &google_protobuf.Empty{}

// Opt is an option for a shim client configuration
type Opt func(context.Context, shim.Config) (shimapi.ShimClient, io.Closer, error)

// WithStart executes a new shim process
func WithStart(binary, address, daemonAddress, cgroup string, nonewns, debug bool, exitHandler func()) Opt {
	return func(ctx context.Context, config shim.Config) (_ shimapi.ShimClient, _ io.Closer, err error) {
		socket, err := newSocket(address)
		if err != nil {
			return nil, nil, err
		}
		defer socket.Close()
		f, err := socket.File()
		if err != nil {
			return nil, nil, errors.Wrapf(err, "failed to get fd for socket %s", address)
		}
		defer f.Close()

		cmd := newCommand(binary, daemonAddress, nonewns, debug, config, f)
		ec, err := reaper.Default.Start(cmd)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "failed to start shim")
		}
		defer func() {
			if err != nil {
				cmd.Process.Kill()
			}
		}()
		go func() {
			reaper.Default.Wait(cmd, ec)
			exitHandler()
		}()
		log.G(ctx).WithFields(logrus.Fields{
			"pid":     cmd.Process.Pid,
			"address": address,
			"debug":   debug,
		}).Infof("shim %s started", binary)
		// set shim in cgroup if it is provided
		if cgroup != "" {
			if err := setCgroup(cgroup, cmd); err != nil {
				return nil, nil, err
			}
			log.G(ctx).WithFields(logrus.Fields{
				"pid":     cmd.Process.Pid,
				"address": address,
			}).Infof("shim placed in cgroup %s", cgroup)
		}
		if err = sys.SetOOMScore(cmd.Process.Pid, sys.OOMScoreMaxKillable); err != nil {
			return nil, nil, errors.Wrap(err, "failed to set OOM Score on shim")
		}
		c, clo, err := WithConnect(address)(ctx, config)
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to connect")
		}
		return c, clo, nil
	}
}

func newCommand(binary, daemonAddress string, nonewns, debug bool, config shim.Config, socket *os.File) *exec.Cmd {
	args := []string{
		"-namespace", config.Namespace,
		"-workdir", config.WorkDir,
		"-address", daemonAddress,
	}

	if config.Criu != "" {
		args = append(args, "-criu-path", config.Criu)
	}
	if config.RuntimeRoot != "" {
		args = append(args, "-runtime-root", config.RuntimeRoot)
	}
	if config.SystemdCgroup {
		args = append(args, "-systemd-cgroup")
	}
	if debug {
		args = append(args, "-debug")
	}

	cmd := exec.Command(binary, args...)
	cmd.Dir = config.Path
	// make sure the shim can be re-parented to system init
	// and is cloned in a new mount namespace because the overlay/filesystems
	// will be mounted by the shim
	cmd.SysProcAttr = getSysProcAttr(nonewns)
	cmd.ExtraFiles = append(cmd.ExtraFiles, socket)
	if debug {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd
}

func newSocket(address string) (*net.UnixListener, error) {
	if len(address) > 106 {
		return nil, errors.Errorf("%q: unix socket path too long (limit 106)", address)
	}
	l, err := net.Listen("unix", "\x00"+address)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to listen to abstract unix socket %q", address)
	}

	return l.(*net.UnixListener), nil
}

func connect(address string, d func(string, time.Duration) (net.Conn, error)) (*grpc.ClientConn, error) {
	gopts := []grpc.DialOption{
		grpc.WithBlock(),
		grpc.WithInsecure(),
		grpc.WithTimeout(100 * time.Second),
		grpc.WithDialer(d),
		grpc.FailOnNonTempDialError(true),
	}
	conn, err := grpc.Dial(dialAddress(address), gopts...)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to dial %q", address)
	}
	return conn, nil
}

func dialer(address string, timeout time.Duration) (net.Conn, error) {
	address = strings.TrimPrefix(address, "unix://")
	return net.DialTimeout("unix", address, timeout)
}

func annonDialer(address string, timeout time.Duration) (net.Conn, error) {
	address = strings.TrimPrefix(address, "unix://")
	return net.DialTimeout("unix", "\x00"+address, timeout)
}

func dialAddress(address string) string {
	return fmt.Sprintf("unix://%s", address)
}

// WithConnect connects to an existing shim
func WithConnect(address string) Opt {
	return func(ctx context.Context, config shim.Config) (shimapi.ShimClient, io.Closer, error) {
		conn, err := connect(address, annonDialer)
		if err != nil {
			return nil, nil, err
		}
		return shimapi.NewShimClient(conn), conn, nil
	}
}

// WithLocal uses an in process shim
func WithLocal(publisher events.Publisher) func(context.Context, shim.Config) (shimapi.ShimClient, io.Closer, error) {
	return func(ctx context.Context, config shim.Config) (shimapi.ShimClient, io.Closer, error) {
		service, err := shim.NewService(config, publisher)
		if err != nil {
			return nil, nil, err
		}
		return shim.NewLocal(service), nil, nil
	}
}

// New returns a new shim client
func New(ctx context.Context, config shim.Config, opt Opt) (*Client, error) {
	s, c, err := opt(ctx, config)
	if err != nil {
		return nil, err
	}
	return &Client{
		ShimClient: s,
		c:          c,
		exitCh:     make(chan struct{}),
	}, nil
}

// Client is a shim client containing the connection to a shim
type Client struct {
	shimapi.ShimClient

	c        io.Closer
	exitCh   chan struct{}
	exitOnce sync.Once
}

// IsAlive returns true if the shim can be contacted.
// NOTE: a negative answer doesn't mean that the process is gone.
func (c *Client) IsAlive(ctx context.Context) (bool, error) {
	_, err := c.ShimInfo(ctx, empty)
	if err != nil {
		if err != grpc.ErrServerStopped {
			return false, err
		}
		return false, nil
	}
	return true, nil
}

// StopShim signals the shim to exit and wait for the process to disappear
func (c *Client) StopShim(ctx context.Context) error {
	return c.signalShim(ctx, unix.SIGTERM)
}

// KillShim kills the shim forcefully and wait for the process to disappear
func (c *Client) KillShim(ctx context.Context) error {
	return c.signalShim(ctx, unix.SIGKILL)
}

// Close the cient connection
func (c *Client) Close() error {
	if c.c == nil {
		return nil
	}
	return c.c.Close()
}

func (c *Client) signalShim(ctx context.Context, sig syscall.Signal) error {
	info, err := c.ShimInfo(ctx, empty)
	if err != nil {
		return err
	}
	pid := int(info.ShimPid)
	// make sure we don't kill ourselves if we are running a local shim
	if os.Getpid() == pid {
		return nil
	}
	if err := unix.Kill(pid, sig); err != nil && err != unix.ESRCH {
		return err
	}
	// wait for shim to die after being signaled
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-c.waitForExit(pid):
		return nil
	}
}

func (c *Client) waitForExit(pid int) <-chan struct{} {
	c.exitOnce.Do(func() {
		for {
			// use kill(pid, 0) here because the shim could have been reparented
			// and we are no longer able to waitpid(pid, ...) on the shim
			if err := unix.Kill(pid, 0); err == unix.ESRCH {
				close(c.exitCh)
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	})
	return c.exitCh
}
