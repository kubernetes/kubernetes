/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dockertools

import (
	"bufio"
	"bytes"
	"fmt"
	"hash/adler32"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/leaky"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/docker/docker/pkg/parsers"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const (
	PodInfraContainerName  = leaky.PodInfraContainerName
	DockerPrefix           = "docker://"
	PodInfraContainerImage = "gcr.io/google_containers/pause:0.8.0"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000
)

// DockerInterface is an abstract interface for testability.  It abstracts the interface of docker.Client.
type DockerInterface interface {
	ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
	RemoveContainer(opts docker.RemoveContainerOptions) error
	InspectImage(image string) (*docker.Image, error)
	ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error)
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
	RemoveImage(image string) error
	Logs(opts docker.LogsOptions) error
	Version() (*docker.Env, error)
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
}

// DockerID is an ID of docker container. It is a type to make it clear when we're working with docker container Ids
type DockerID string

type KubeletContainerName struct {
	PodFullName   string
	PodUID        types.UID
	ContainerName string
}

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string) error
	IsImagePresent(image string) (bool, error)
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client  DockerInterface
	keyring credentialprovider.DockerKeyring
}

type throttledDockerPuller struct {
	puller  dockerPuller
	limiter util.RateLimiter
}

// newDockerPuller creates a new instance of the default implementation of DockerPuller.
func newDockerPuller(client DockerInterface, qps float32, burst int) DockerPuller {
	dp := dockerPuller{
		client:  client,
		keyring: credentialprovider.NewDockerKeyring(),
	}

	if qps == 0.0 {
		return dp
	}
	return &throttledDockerPuller{
		puller:  dp,
		limiter: util.NewTokenBucketRateLimiter(qps, burst),
	}
}

type dockerContainerCommandRunner struct {
	client DockerInterface
}

// The first version of docker that supports exec natively is 1.3.0 == API 1.15
var dockerAPIVersionWithExec, _ = docker.NewAPIVersion("1.15")

// Returns the major and minor version numbers of docker server.
// TODO(yifan): Remove this once the ContainerCommandRunner is implemented by dockerManager.
func (d *dockerContainerCommandRunner) getDockerServerVersion() (docker.APIVersion, error) {
	env, err := d.client.Version()
	if err != nil {
		return nil, fmt.Errorf("failed to get docker server version - %v", err)
	}

	apiVersion := env.Get("ApiVersion")
	version, err := docker.NewAPIVersion(apiVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to parse docker server version %q: %v", apiVersion, err)
	}

	return version, nil
}

func (d *dockerContainerCommandRunner) nativeExecSupportExists() (bool, error) {
	version, err := d.getDockerServerVersion()
	if err != nil {
		return false, err
	}
	return version.GreaterThanOrEqualTo(dockerAPIVersionWithExec), nil
}

func (d *dockerContainerCommandRunner) getRunInContainerCommand(containerID string, cmd []string) (*exec.Cmd, error) {
	args := append([]string{"exec"}, cmd...)
	command := exec.Command("/usr/sbin/nsinit", args...)
	command.Dir = fmt.Sprintf("/var/lib/docker/execdriver/native/%s", containerID)
	return command, nil
}

func (d *dockerContainerCommandRunner) runInContainerUsingNsinit(containerID string, cmd []string) ([]byte, error) {
	c, err := d.getRunInContainerCommand(containerID, cmd)
	if err != nil {
		return nil, err
	}
	return c.CombinedOutput()
}

// RunInContainer uses nsinit to run the command inside the container identified by containerID
// TODO(yifan): Use strong type for containerID.
func (d *dockerContainerCommandRunner) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	// If native exec support does not exist in the local docker daemon use nsinit.
	useNativeExec, err := d.nativeExecSupportExists()
	if err != nil {
		return nil, err
	}
	if !useNativeExec {
		return d.runInContainerUsingNsinit(containerID, cmd)
	}
	createOpts := docker.CreateExecOptions{
		Container:    containerID,
		Cmd:          cmd,
		AttachStdin:  false,
		AttachStdout: true,
		AttachStderr: true,
		Tty:          false,
	}
	execObj, err := d.client.CreateExec(createOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to run in container - Exec setup failed - %v", err)
	}
	var buf bytes.Buffer
	wrBuf := bufio.NewWriter(&buf)
	startOpts := docker.StartExecOptions{
		Detach:       false,
		Tty:          false,
		OutputStream: wrBuf,
		ErrorStream:  wrBuf,
		RawTerminal:  false,
	}
	errChan := make(chan error, 1)
	go func() {
		errChan <- d.client.StartExec(execObj.ID, startOpts)
	}()
	wrBuf.Flush()
	return buf.Bytes(), <-errChan
}

// ExecInContainer uses nsenter to run the command inside the container identified by containerID.
//
// TODO:
//  - match cgroups of container
//  - should we support `docker exec`?
//  - should we support nsenter in a container, running with elevated privs and --pid=host?
//  - use strong type for containerId
func (d *dockerContainerCommandRunner) ExecInContainer(containerId string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	nsenter, err := exec.LookPath("nsenter")
	if err != nil {
		return fmt.Errorf("exec unavailable - unable to locate nsenter")
	}

	container, err := d.client.InspectContainer(containerId)
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container)
	}

	containerPid := container.State.Pid

	// TODO what if the container doesn't have `env`???
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-m", "-i", "-u", "-n", "-p", "--", "env", "-i"}
	args = append(args, fmt.Sprintf("HOSTNAME=%s", container.Config.Hostname))
	args = append(args, container.Config.Env...)
	args = append(args, cmd...)
	command := exec.Command(nsenter, args...)
	if tty {
		p, err := StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		if stdin != nil {
			go io.Copy(p, stdin)
		}

		if stdout != nil {
			go io.Copy(stdout, p)
		}

		return command.Wait()
	} else {
		if stdin != nil {
			// Use an os.Pipe here as it returns true *os.File objects.
			// This way, if you run 'kubectl exec -p <pod> -i bash' (no tty) and type 'exit',
			// the call below to command.Run() can unblock because its Stdin is the read half
			// of the pipe.
			r, w, err := os.Pipe()
			if err != nil {
				return err
			}
			go io.Copy(w, stdin)

			command.Stdin = r
		}
		if stdout != nil {
			command.Stdout = stdout
		}
		if stderr != nil {
			command.Stderr = stderr
		}

		return command.Run()
	}
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
func (d *dockerContainerCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	podInfraContainer := pod.FindContainerByName(PodInfraContainerName)
	if podInfraContainer == nil {
		return fmt.Errorf("cannot find pod infra container in pod %q", kubecontainer.BuildPodFullName(pod.Name, pod.Namespace))
	}
	container, err := d.client.InspectContainer(string(podInfraContainer.ID))
	if err != nil {
		return err
	}

	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container)
	}

	containerPid := container.State.Pid
	// TODO what if the host doesn't have it???
	_, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("Unable to do port forwarding: socat not found.")
	}
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-n", "socat", "-", fmt.Sprintf("TCP4:localhost:%d", port)}
	// TODO use exec.LookPath
	command := exec.Command("nsenter", args...)
	command.Stdin = stream
	command.Stdout = stream
	return command.Run()
}

// NewDockerContainerCommandRunner creates a ContainerCommandRunner which uses nsinit to run a command
// inside a container.
func NewDockerContainerCommandRunner(client DockerInterface) ContainerCommandRunner {
	return &dockerContainerCommandRunner{client: client}
}

func parseImageName(image string) (string, string) {
	return parsers.ParseRepositoryTag(image)
}

func (p dockerPuller) Pull(image string) error {
	repoToPull, tag := parseImageName(image)

	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = "latest"
	}

	opts := docker.PullImageOptions{
		Repository: repoToPull,
		Tag:        tag,
	}

	creds, ok := p.keyring.Lookup(repoToPull)
	if !ok {
		glog.V(1).Infof("Pulling image %s without credentials", image)
	}

	err := p.client.PullImage(opts, creds)
	// If there was no error, or we had credentials, just return the error.
	if err == nil || ok {
		return err
	}
	// Image spec: [<registry>/]<repository>/<image>[:<version] so we count '/'
	explicitRegistry := (strings.Count(image, "/") == 2)
	// Hack, look for a private registry, and decorate the error with the lack of
	// credentials.  This is heuristic, and really probably could be done better
	// by talking to the registry API directly from the kubelet here.
	if explicitRegistry {
		return fmt.Errorf("image pull failed for %s, this may be because there are no credentials on this request.  details: (%v)", image, err)
	}
	return err
}

func (p throttledDockerPuller) Pull(image string) error {
	if p.limiter.CanAccept() {
		return p.puller.Pull(image)
	}
	return fmt.Errorf("pull QPS exceeded.")
}

func (p dockerPuller) IsImagePresent(image string) (bool, error) {
	_, err := p.client.InspectImage(image)
	if err == nil {
		return true, nil
	}
	// This is super brittle, but its the best we got.
	// TODO: Land code in the docker client to use docker.Error here instead.
	if err.Error() == "no such image" {
		return false, nil
	}
	return false, err
}

func (p throttledDockerPuller) IsImagePresent(name string) (bool, error) {
	return p.puller.IsImagePresent(name)
}

// DockerContainers is a map of containers
type DockerContainers map[DockerID]*docker.APIContainers

func (c DockerContainers) FindPodContainer(podFullName string, uid types.UID, containerName string) (*docker.APIContainers, bool, uint64) {
	for _, dockerContainer := range c {
		if len(dockerContainer.Names) == 0 {
			continue
		}
		// TODO(proppy): build the docker container name and do a map lookup instead?
		dockerName, hash, err := ParseDockerName(dockerContainer.Names[0])
		if err != nil {
			continue
		}
		if dockerName.PodFullName == podFullName &&
			(uid == "" || dockerName.PodUID == uid) &&
			dockerName.ContainerName == containerName {
			return dockerContainer, true, hash
		}
	}
	return nil, false, 0
}

const containerNamePrefix = "k8s"

func HashContainer(container *api.Container) uint64 {
	hash := adler32.New()
	util.DeepHashObject(hash, *container)
	return uint64(hash.Sum32())
}

// Creates a name which can be reversed to identify both full pod name and container name.
func BuildDockerName(dockerName KubeletContainerName, container *api.Container) string {
	containerName := dockerName.ContainerName + "." + strconv.FormatUint(HashContainer(container), 16)
	return fmt.Sprintf("%s_%s_%s_%s_%08x",
		containerNamePrefix,
		containerName,
		dockerName.PodFullName,
		dockerName.PodUID,
		rand.Uint32())
}

// Unpacks a container name, returning the pod full name and container name we would have used to
// construct the docker name. If we are unable to parse the name, an error is returned.
func ParseDockerName(name string) (dockerName *KubeletContainerName, hash uint64, err error) {
	// For some reason docker appears to be appending '/' to names.
	// If it's there, strip it.
	name = strings.TrimPrefix(name, "/")
	parts := strings.Split(name, "_")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		err = fmt.Errorf("failed to parse Docker container name %q into parts", name)
		return nil, 0, err
	}
	if len(parts) < 6 {
		// We have at least 5 fields.  We may have more in the future.
		// Anything with less fields than this is not something we can
		// manage.
		glog.Warningf("found a container with the %q prefix, but too few fields (%d): %q", containerNamePrefix, len(parts), name)
		err = fmt.Errorf("Docker container name %q has less parts than expected %v", name, parts)
		return nil, 0, err
	}

	nameParts := strings.Split(parts[1], ".")
	containerName := nameParts[0]
	if len(nameParts) > 1 {
		hash, err = strconv.ParseUint(nameParts[1], 16, 32)
		if err != nil {
			glog.Warningf("invalid container hash %q in container %q", nameParts[1], name)
		}
	}

	podFullName := parts[2] + "_" + parts[3]
	podUID := types.UID(parts[4])

	return &KubeletContainerName{podFullName, podUID, containerName}, hash, nil
}

// Get a docker endpoint, either from the string passed in, or $DOCKER_HOST environment variables
func getDockerEndpoint(dockerEndpoint string) string {
	var endpoint string
	if len(dockerEndpoint) > 0 {
		endpoint = dockerEndpoint
	} else if len(os.Getenv("DOCKER_HOST")) > 0 {
		endpoint = os.Getenv("DOCKER_HOST")
	} else {
		endpoint = "unix:///var/run/docker.sock"
	}
	glog.Infof("Connecting to docker on %s", endpoint)

	return endpoint
}

func ConnectToDockerOrDie(dockerEndpoint string) DockerInterface {
	if dockerEndpoint == "fake://" {
		return &FakeDockerClient{
			VersionInfo: []string{"apiVersion=1.16"},
		}
	}
	client, err := docker.NewClient(getDockerEndpoint(dockerEndpoint))
	if err != nil {
		glog.Fatal("Couldn't connect to docker.")
	}
	return client
}

// TODO(yifan): Move this to container.Runtime.
type ContainerCommandRunner interface {
	RunInContainer(containerID string, cmd []string) ([]byte, error)
	ExecInContainer(containerID string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error
}

func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// zero milliCPU means unset. Use kernel default.
		return 0
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

// GetKubeletDockerContainers lists all container or just the running ones.
// Returns a map of docker containers that we manage, keyed by container ID.
// TODO: Move this function with dockerCache to DockerManager.
func GetKubeletDockerContainers(client DockerInterface, allContainers bool) (DockerContainers, error) {
	result := make(DockerContainers)
	containers, err := client.ListContainers(docker.ListContainersOptions{All: allContainers})
	if err != nil {
		return nil, err
	}
	for i := range containers {
		container := &containers[i]
		if len(container.Names) == 0 {
			continue
		}
		// Skip containers that we didn't create to allow users to manually
		// spin up their own containers if they want.
		// TODO(dchen1107): Remove the old separator "--" by end of Oct
		if !strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"_") &&
			!strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"--") {
			glog.V(3).Infof("Docker Container: %s is not managed by kubelet.", container.Names[0])
			continue
		}
		result[DockerID(container.ID)] = container
	}
	return result, nil
}
