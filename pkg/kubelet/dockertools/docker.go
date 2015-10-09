/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"math/rand"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/parsers"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
)

const (
	PodInfraContainerName  = leaky.PodInfraContainerName
	DockerPrefix           = "docker://"
	PodInfraContainerImage = "gcr.io/google_containers/pause:0.8.0"
	LogSuffix              = "log"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	quotaPeriod = 100000
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
	Info() (*docker.Env, error)
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
	InspectExec(id string) (*docker.ExecInspect, error)
	AttachToContainer(opts docker.AttachToContainerOptions) error
}

// KubeletContainerName encapsulates a pod name and a Kubernetes container name.
type KubeletContainerName struct {
	PodFullName   string
	PodUID        types.UID
	ContainerName string
}

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string, secrets []api.Secret) error
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

func parseImageName(image string) (string, string) {
	return parsers.ParseRepositoryTag(image)
}

func filterHTTPError(err error, image string) error {
	// docker/docker/pull/11314 prints detailed error info for docker pull.
	// When it hits 502, it returns a verbose html output including an inline svg,
	// which makes the output of kubectl get pods much harder to parse.
	// Here converts such verbose output to a concise one.
	jerr, ok := err.(*jsonmessage.JSONError)
	if ok && (jerr.Code == http.StatusBadGateway ||
		jerr.Code == http.StatusServiceUnavailable ||
		jerr.Code == http.StatusGatewayTimeout) {
		glog.V(2).Infof("Pulling image %q failed: %v", image, err)
		return fmt.Errorf("image pull failed for %s because the registry is temporarily unavailable.", image)
	} else {
		return err
	}
}

func (p dockerPuller) Pull(image string, secrets []api.Secret) error {
	repoToPull, tag := parseImageName(image)

	// If no tag was specified, use the default "latest".
	if len(tag) == 0 {
		tag = "latest"
	}

	opts := docker.PullImageOptions{
		Repository: repoToPull,
		Tag:        tag,
	}

	keyring, err := credentialprovider.MakeDockerKeyring(secrets, p.keyring)
	if err != nil {
		return err
	}

	creds, haveCredentials := keyring.Lookup(repoToPull)
	if !haveCredentials {
		glog.V(1).Infof("Pulling image %s without credentials", image)

		err := p.client.PullImage(opts, docker.AuthConfiguration{})
		if err == nil {
			return nil
		}

		// Image spec: [<registry>/]<repository>/<image>[:<version] so we count '/'
		explicitRegistry := (strings.Count(image, "/") == 2)
		// Hack, look for a private registry, and decorate the error with the lack of
		// credentials.  This is heuristic, and really probably could be done better
		// by talking to the registry API directly from the kubelet here.
		if explicitRegistry {
			return fmt.Errorf("image pull failed for %s, this may be because there are no credentials on this request.  details: (%v)", image, err)
		}

		return filterHTTPError(err, image)
	}

	var pullErrs []error
	for _, currentCreds := range creds {
		err := p.client.PullImage(opts, currentCreds)
		// If there was no error, return success
		if err == nil {
			return nil
		}

		pullErrs = append(pullErrs, filterHTTPError(err, image))
	}

	return utilerrors.NewAggregate(pullErrs)
}

func (p throttledDockerPuller) Pull(image string, secrets []api.Secret) error {
	if p.limiter.CanAccept() {
		return p.puller.Pull(image, secrets)
	}
	return fmt.Errorf("pull QPS exceeded.")
}

func (p dockerPuller) IsImagePresent(image string) (bool, error) {
	_, err := p.client.InspectImage(image)
	if err == nil {
		return true, nil
	}
	if err == docker.ErrNoSuchImage {
		return false, nil
	}
	return false, err
}

func (p throttledDockerPuller) IsImagePresent(name string) (bool, error) {
	return p.puller.IsImagePresent(name)
}

// DockerContainers is a map of containers
type DockerContainers map[kubetypes.DockerID]*docker.APIContainers

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

// Creates a name which can be reversed to identify both full pod name and container name.
func BuildDockerName(dockerName KubeletContainerName, container *api.Container) (string, string) {
	containerName := dockerName.ContainerName + "." + strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	stableName := fmt.Sprintf("%s_%s_%s_%s",
		containerNamePrefix,
		containerName,
		dockerName.PodFullName,
		dockerName.PodUID)

	return stableName, fmt.Sprintf("%s_%08x", stableName, rand.Uint32())
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

func LogSymlink(containerLogsDir, podFullName, containerName, dockerId string) string {
	return path.Join(containerLogsDir, fmt.Sprintf("%s_%s-%s.%s", podFullName, containerName, dockerId, LogSuffix))
}

// Get a *docker.Client, either using the endpoint passed in, or using
// DOCKER_HOST, DOCKER_TLS_VERIFY, and DOCKER_CERT path per their spec
func getDockerClient(dockerEndpoint string) (*docker.Client, error) {
	if len(dockerEndpoint) > 0 {
		glog.Infof("Connecting to docker on %s", dockerEndpoint)
		return docker.NewClient(dockerEndpoint)
	}
	return docker.NewClientFromEnv()
}

func ConnectToDockerOrDie(dockerEndpoint string) DockerInterface {
	if dockerEndpoint == "fake://" {
		return &FakeDockerClient{
			VersionInfo: docker.Env{"ApiVersion=1.18"},
		}
	}
	client, err := getDockerClient(dockerEndpoint)
	if err != nil {
		glog.Fatalf("Couldn't connect to docker: %v", err)
	}
	return client
}

// milliCPUToQuota converts milliCPU to CFS quota and period values
func milliCPUToQuota(milliCPU int64) (quota int64, period int64) {
	// CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts

	if milliCPU == 0 {
		// take the default behavior from docker
		return
	}

	// we set the period to 100ms by default
	period = quotaPeriod

	// we then convert your milliCPU to a value normalized over a period
	quota = (milliCPU * quotaPeriod) / milliCPUToCPU

	return
}

func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// Docker converts zero milliCPU to unset, which maps to kernel default
		// for unset: 1024. Return 2 here to really match kernel default for
		// zero milliCPU.
		return minShares
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
		result[kubetypes.DockerID(container.ID)] = container
	}
	return result, nil
}
