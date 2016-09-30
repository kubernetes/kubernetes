/*
Copyright 2014 The Kubernetes Authors.

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
	"time"

	dockerdigest "github.com/docker/distribution/digest"
	dockerref "github.com/docker/distribution/reference"
	"github.com/docker/docker/pkg/jsonmessage"
	dockerapi "github.com/docker/engine-api/client"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/parsers"
)

const (
	PodInfraContainerName = leaky.PodInfraContainerName
	DockerPrefix          = "docker://"
	DockerPullablePrefix  = "docker-pullable://"
	LogSuffix             = "log"
	ext4MaxFileNameLen    = 255
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	quotaPeriod    = 100000
	minQuotaPeriod = 1000
)

// DockerInterface is an abstract interface for testability.  It abstracts the interface of docker client.
type DockerInterface interface {
	ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error)
	InspectContainer(id string) (*dockertypes.ContainerJSON, error)
	CreateContainer(dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error)
	StartContainer(id string) error
	StopContainer(id string, timeout int) error
	RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error
	InspectImageByRef(imageRef string) (*dockertypes.ImageInspect, error)
	InspectImageByID(imageID string) (*dockertypes.ImageInspect, error)
	ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error)
	PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error
	RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error)
	ImageHistory(id string) ([]dockertypes.ImageHistory, error)
	Logs(string, dockertypes.ContainerLogsOptions, StreamOptions) error
	Version() (*dockertypes.Version, error)
	Info() (*dockertypes.Info, error)
	CreateExec(string, dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error)
	StartExec(string, dockertypes.ExecStartCheck, StreamOptions) error
	InspectExec(id string) (*dockertypes.ContainerExecInspect, error)
	AttachToContainer(string, dockertypes.ContainerAttachOptions, StreamOptions) error
	ResizeContainerTTY(id string, height, width int) error
	ResizeExecTTY(id string, height, width int) error
}

// KubeletContainerName encapsulates a pod name and a Kubernetes container name.
type KubeletContainerName struct {
	PodFullName   string
	PodUID        types.UID
	ContainerName string
}

// containerNamePrefix is used to identify the containers on the node managed by this
// process.
var containerNamePrefix = "k8s"

// SetContainerNamePrefix allows the container prefix name for this process to be changed.
// This is intended to support testing and bootstrapping experimentation. It cannot be
// changed once the Kubelet starts.
func SetContainerNamePrefix(prefix string) {
	containerNamePrefix = prefix
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
	limiter flowcontrol.RateLimiter
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
		limiter: flowcontrol.NewTokenBucketRateLimiter(qps, burst),
	}
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
		return images.RegistryUnavailable
	} else {
		return err
	}
}

// matchImageTagOrSHA checks if the given image specifier is a valid image ref,
// and that it matches the given image. It should fail on things like image IDs
// (config digests) and other digest-only references, but succeed on image names
// (`foo`), tag references (`foo:bar`), and manifest digest references
// (`foo@sha256:xyz`).
func matchImageTagOrSHA(inspected dockertypes.ImageInspect, image string) bool {
	// The image string follows the grammar specified here
	// https://github.com/docker/distribution/blob/master/reference/reference.go#L4
	named, err := dockerref.ParseNamed(image)
	if err != nil {
		glog.V(4).Infof("couldn't parse image reference %q: %v", image, err)
		return false
	}
	_, isTagged := named.(dockerref.Tagged)
	digest, isDigested := named.(dockerref.Digested)
	if !isTagged && !isDigested {
		// No Tag or SHA specified, so just return what we have
		return true
	}

	if isTagged {
		// Check the RepoTags for a match.
		for _, tag := range inspected.RepoTags {
			// An image name (without the tag/digest) can be [hostname '/'] component ['/' component]*
			// Because either the RepoTag or the name *may* contain the
			// hostname or not, we only check for the suffix match.
			if strings.HasSuffix(image, tag) || strings.HasSuffix(tag, image) {
				return true
			}
		}
	}

	if isDigested {
		for _, repoDigest := range inspected.RepoDigests {
			named, err := dockerref.ParseNamed(repoDigest)
			if err != nil {
				glog.V(4).Infof("couldn't parse image RepoDigest reference %q: %v", repoDigest, err)
				continue
			}
			if d, isDigested := named.(dockerref.Digested); isDigested {
				if digest.Digest().Algorithm().String() == d.Digest().Algorithm().String() &&
					digest.Digest().Hex() == d.Digest().Hex() {
					return true
				}
			}
		}

		// process the ID as a digest
		id, err := dockerdigest.ParseDigest(inspected.ID)
		if err != nil {
			glog.V(4).Infof("couldn't parse image ID reference %q: %v", id, err)
			return false
		}
		if digest.Digest().Algorithm().String() == id.Algorithm().String() && digest.Digest().Hex() == id.Hex() {
			return true
		}
	}
	glog.V(4).Infof("Inspected image (%q) does not match %s", inspected.ID, image)
	return false
}

// applyDefaultImageTag parses a docker image string, if it doesn't contain any tag or digest,
// a default tag will be applied.
func applyDefaultImageTag(image string) (string, error) {
	named, err := dockerref.ParseNamed(image)
	if err != nil {
		return "", fmt.Errorf("couldn't parse image reference %q: %v", image, err)
	}
	_, isTagged := named.(dockerref.Tagged)
	_, isDigested := named.(dockerref.Digested)
	if !isTagged && !isDigested {
		named, err := dockerref.WithTag(named, parsers.DefaultImageTag)
		if err != nil {
			return "", fmt.Errorf("failed to apply default image tag %q: %v", image, err)
		}
		image = named.String()
	}
	return image, nil
}

// matchImageIDOnly checks that the given image specifier is a digest-only
// reference, and that it matches the given image.
func matchImageIDOnly(inspected dockertypes.ImageInspect, image string) bool {
	// If the image ref is literally equal to the inspected image's ID,
	// just return true here (this might be the case for Docker 1.9,
	// where we won't have a digest for the ID)
	if inspected.ID == image {
		return true
	}

	// Otherwise, we should try actual parsing to be more correct
	ref, err := dockerref.Parse(image)
	if err != nil {
		glog.V(4).Infof("couldn't parse image reference %q: %v", image, err)
		return false
	}

	digest, isDigested := ref.(dockerref.Digested)
	if !isDigested {
		glog.V(4).Infof("the image reference %q was not a digest reference")
		return false
	}

	id, err := dockerdigest.ParseDigest(inspected.ID)
	if err != nil {
		glog.V(4).Infof("couldn't parse image ID reference %q: %v", id, err)
		return false
	}

	if digest.Digest().Algorithm().String() == id.Algorithm().String() && digest.Digest().Hex() == id.Hex() {
		return true
	}

	glog.V(4).Infof("The reference %s does not directly refer to the given image's ID (%q)", image, inspected.ID)
	return false
}

func (p dockerPuller) Pull(image string, secrets []api.Secret) error {
	// If the image contains no tag or digest, a default tag should be applied.
	image, err := applyDefaultImageTag(image)
	if err != nil {
		return err
	}

	keyring, err := credentialprovider.MakeDockerKeyring(secrets, p.keyring)
	if err != nil {
		return err
	}

	// The only used image pull option RegistryAuth will be set in kube_docker_client
	opts := dockertypes.ImagePullOptions{}

	creds, haveCredentials := keyring.Lookup(image)
	if !haveCredentials {
		glog.V(1).Infof("Pulling image %s without credentials", image)

		err := p.client.PullImage(image, dockertypes.AuthConfig{}, opts)
		if err == nil {
			// Sometimes PullImage failed with no error returned.
			exist, ierr := p.IsImagePresent(image)
			if ierr != nil {
				glog.Warningf("Failed to inspect image %s: %v", image, ierr)
			}
			if !exist {
				return fmt.Errorf("image pull failed for unknown error")
			}
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
		err = p.client.PullImage(image, credentialprovider.LazyProvide(currentCreds), opts)
		// If there was no error, return success
		if err == nil {
			return nil
		}

		pullErrs = append(pullErrs, filterHTTPError(err, image))
	}

	return utilerrors.NewAggregate(pullErrs)
}

func (p throttledDockerPuller) Pull(image string, secrets []api.Secret) error {
	if p.limiter.TryAccept() {
		return p.puller.Pull(image, secrets)
	}
	return fmt.Errorf("pull QPS exceeded.")
}

func (p dockerPuller) IsImagePresent(image string) (bool, error) {
	_, err := p.client.InspectImageByRef(image)
	if err == nil {
		return true, nil
	}
	if _, ok := err.(imageNotFoundError); ok {
		return false, nil
	}
	return false, err
}

func (p throttledDockerPuller) IsImagePresent(name string) (bool, error) {
	return p.puller.IsImagePresent(name)
}

// Creates a name which can be reversed to identify both full pod name and container name.
// This function returns stable name, unique name and a unique id.
// Although rand.Uint32() is not really unique, but it's enough for us because error will
// only occur when instances of the same container in the same pod have the same UID. The
// chance is really slim.
func BuildDockerName(dockerName KubeletContainerName, container *api.Container) (string, string, string) {
	containerName := dockerName.ContainerName + "." + strconv.FormatUint(kubecontainer.HashContainer(container), 16)
	stableName := fmt.Sprintf("%s_%s_%s_%s",
		containerNamePrefix,
		containerName,
		dockerName.PodFullName,
		dockerName.PodUID)
	UID := fmt.Sprintf("%08x", rand.Uint32())
	return stableName, fmt.Sprintf("%s_%s", stableName, UID), UID
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
	suffix := fmt.Sprintf(".%s", LogSuffix)
	logPath := fmt.Sprintf("%s_%s-%s", podFullName, containerName, dockerId)
	// Length of a filename cannot exceed 255 characters in ext4 on Linux.
	if len(logPath) > ext4MaxFileNameLen-len(suffix) {
		logPath = logPath[:ext4MaxFileNameLen-len(suffix)]
	}
	return path.Join(containerLogsDir, logPath+suffix)
}

// Get a *dockerapi.Client, either using the endpoint passed in, or using
// DOCKER_HOST, DOCKER_TLS_VERIFY, and DOCKER_CERT path per their spec
func getDockerClient(dockerEndpoint string) (*dockerapi.Client, error) {
	if len(dockerEndpoint) > 0 {
		glog.Infof("Connecting to docker on %s", dockerEndpoint)
		return dockerapi.NewClient(dockerEndpoint, "", nil, nil)
	}
	return dockerapi.NewEnvClient()
}

// ConnectToDockerOrDie creates docker client connecting to docker daemon.
// If the endpoint passed in is "fake://", a fake docker client
// will be returned. The program exits if error occurs. The requestTimeout
// is the timeout for docker requests. If timeout is exceeded, the request
// will be cancelled and throw out an error. If requestTimeout is 0, a default
// value will be applied.
func ConnectToDockerOrDie(dockerEndpoint string, requestTimeout time.Duration) DockerInterface {
	if dockerEndpoint == "fake://" {
		return NewFakeDockerClient()
	}
	client, err := getDockerClient(dockerEndpoint)
	if err != nil {
		glog.Fatalf("Couldn't connect to docker: %v", err)
	}
	glog.Infof("Start docker client with request timeout=%v", requestTimeout)
	return newKubeDockerClient(client, requestTimeout)
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

	// quota needs to be a minimum of 1ms.
	if quota < minQuotaPeriod {
		quota = minQuotaPeriod
	}

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
// Returns a list of docker containers that we manage
func GetKubeletDockerContainers(client DockerInterface, allContainers bool) ([]*dockertypes.Container, error) {
	result := []*dockertypes.Container{}
	containers, err := client.ListContainers(dockertypes.ContainerListOptions{All: allContainers})
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
		if !strings.HasPrefix(container.Names[0], "/"+containerNamePrefix+"_") {
			glog.V(5).Infof("Docker Container: %s is not managed by kubelet.", container.Names[0])
			continue
		}
		result = append(result, container)
	}
	return result, nil
}
