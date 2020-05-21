// +build !dockerless

/*
Copyright 2016 The Kubernetes Authors.

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

package dockershim

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"

	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	dockerfilters "github.com/docker/docker/api/types/filters"
	dockernat "github.com/docker/go-connections/nat"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/parsers"
)

const (
	annotationPrefix     = "annotation."
	securityOptSeparator = '='
)

var (
	conflictRE = regexp.MustCompile(`Conflict. (?:.)+ is already in use by container \"?([0-9a-z]+)\"?`)

	// this is hacky, but extremely common.
	// if a container starts but the executable file is not found, runc gives a message that matches
	startRE = regexp.MustCompile(`\\\\\\\"(.*)\\\\\\\": executable file not found`)

	defaultSeccompOpt = []dockerOpt{{"seccomp", v1.SeccompProfileNameUnconfined, ""}}
)

// generateEnvList converts KeyValue list to a list of strings, in the form of
// '<key>=<value>', which can be understood by docker.
func generateEnvList(envs []*runtimeapi.KeyValue) (result []string) {
	for _, env := range envs {
		result = append(result, fmt.Sprintf("%s=%s", env.Key, env.Value))
	}
	return
}

// makeLabels converts annotations to labels and merge them with the given
// labels. This is necessary because docker does not support annotations;
// we *fake* annotations using labels. Note that docker labels are not
// updatable.
func makeLabels(labels, annotations map[string]string) map[string]string {
	merged := make(map[string]string)
	for k, v := range labels {
		merged[k] = v
	}
	for k, v := range annotations {
		// Assume there won't be conflict.
		merged[fmt.Sprintf("%s%s", annotationPrefix, k)] = v
	}
	return merged
}

// extractLabels converts raw docker labels to the CRI labels and annotations.
// It also filters out internal labels used by this shim.
func extractLabels(input map[string]string) (map[string]string, map[string]string) {
	labels := make(map[string]string)
	annotations := make(map[string]string)
	for k, v := range input {
		// Check if the key is used internally by the shim.
		internal := false
		for _, internalKey := range internalLabelKeys {
			if k == internalKey {
				internal = true
				break
			}
		}
		if internal {
			continue
		}

		// Delete the container name label for the sandbox. It is added in the shim,
		// should not be exposed via CRI.
		if k == types.KubernetesContainerNameLabel &&
			input[containerTypeLabelKey] == containerTypeLabelSandbox {
			continue
		}

		// Check if the label should be treated as an annotation.
		if strings.HasPrefix(k, annotationPrefix) {
			annotations[strings.TrimPrefix(k, annotationPrefix)] = v
			continue
		}
		labels[k] = v
	}
	return labels, annotations
}

// generateMountBindings converts the mount list to a list of strings that
// can be understood by docker.
// '<HostPath>:<ContainerPath>[:options]', where 'options'
// is a comma-separated list of the following strings:
// 'ro', if the path is read only
// 'Z', if the volume requires SELinux relabeling
// propagation mode such as 'rslave'
func generateMountBindings(mounts []*runtimeapi.Mount) []string {
	result := make([]string, 0, len(mounts))
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.HostPath, m.ContainerPath)
		var attrs []string
		if m.Readonly {
			attrs = append(attrs, "ro")
		}
		// Only request relabeling if the pod provides an SELinux context. If the pod
		// does not provide an SELinux context relabeling will label the volume with
		// the container's randomly allocated MCS label. This would restrict access
		// to the volume to the container which mounts it first.
		if m.SelinuxRelabel {
			attrs = append(attrs, "Z")
		}
		switch m.Propagation {
		case runtimeapi.MountPropagation_PROPAGATION_PRIVATE:
			// noop, private is default
		case runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL:
			attrs = append(attrs, "rshared")
		case runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER:
			attrs = append(attrs, "rslave")
		default:
			klog.Warningf("unknown propagation mode for hostPath %q", m.HostPath)
			// Falls back to "private"
		}

		if len(attrs) > 0 {
			bind = fmt.Sprintf("%s:%s", bind, strings.Join(attrs, ","))
		}
		result = append(result, bind)
	}
	return result
}

func makePortsAndBindings(pm []*runtimeapi.PortMapping) (dockernat.PortSet, map[dockernat.Port][]dockernat.PortBinding) {
	exposedPorts := dockernat.PortSet{}
	portBindings := map[dockernat.Port][]dockernat.PortBinding{}
	for _, port := range pm {
		exteriorPort := port.HostPort
		if exteriorPort == 0 {
			// No need to do port binding when HostPort is not specified
			continue
		}
		interiorPort := port.ContainerPort
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		var protocol string
		switch port.Protocol {
		case runtimeapi.Protocol_UDP:
			protocol = "/udp"
		case runtimeapi.Protocol_TCP:
			protocol = "/tcp"
		case runtimeapi.Protocol_SCTP:
			protocol = "/sctp"
		default:
			klog.Warningf("Unknown protocol %q: defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}

		dockerPort := dockernat.Port(strconv.Itoa(int(interiorPort)) + protocol)
		exposedPorts[dockerPort] = struct{}{}

		hostBinding := dockernat.PortBinding{
			HostPort: strconv.Itoa(int(exteriorPort)),
			HostIP:   port.HostIp,
		}

		// Allow multiple host ports bind to same docker port
		if existedBindings, ok := portBindings[dockerPort]; ok {
			// If a docker port already map to a host port, just append the host ports
			portBindings[dockerPort] = append(existedBindings, hostBinding)
		} else {
			// Otherwise, it's fresh new port binding
			portBindings[dockerPort] = []dockernat.PortBinding{
				hostBinding,
			}
		}
	}
	return exposedPorts, portBindings
}

// getApparmorSecurityOpts gets apparmor options from container config.
func getApparmorSecurityOpts(sc *runtimeapi.LinuxContainerSecurityContext, separator rune) ([]string, error) {
	if sc == nil || sc.ApparmorProfile == "" {
		return nil, nil
	}

	appArmorOpts, err := getAppArmorOpts(sc.ApparmorProfile)
	if err != nil {
		return nil, err
	}

	fmtOpts := fmtDockerOpts(appArmorOpts, separator)
	return fmtOpts, nil
}

// dockerFilter wraps around dockerfilters.Args and provides methods to modify
// the filter easily.
type dockerFilter struct {
	args *dockerfilters.Args
}

func newDockerFilter(args *dockerfilters.Args) *dockerFilter {
	return &dockerFilter{args: args}
}

func (f *dockerFilter) Add(key, value string) {
	f.args.Add(key, value)
}

func (f *dockerFilter) AddLabel(key, value string) {
	f.Add("label", fmt.Sprintf("%s=%s", key, value))
}

// parseUserFromImageUser splits the user out of an user:group string.
func parseUserFromImageUser(id string) string {
	if id == "" {
		return id
	}
	// split instances where the id may contain user:group
	if strings.Contains(id, ":") {
		return strings.Split(id, ":")[0]
	}
	// no group, just return the id
	return id
}

// getUserFromImageUser gets uid or user name of the image user.
// If user is numeric, it will be treated as uid; or else, it is treated as user name.
func getUserFromImageUser(imageUser string) (*int64, string) {
	user := parseUserFromImageUser(imageUser)
	// return both nil if user is not specified in the image.
	if user == "" {
		return nil, ""
	}
	// user could be either uid or user name. Try to interpret as numeric uid.
	uid, err := strconv.ParseInt(user, 10, 64)
	if err != nil {
		// If user is non numeric, assume it's user name.
		return nil, user
	}
	// If user is a numeric uid.
	return &uid, ""
}

// See #33189. If the previous attempt to create a sandbox container name FOO
// failed due to "device or resource busy", it is possible that docker did
// not clean up properly and has inconsistent internal state. Docker would
// not report the existence of FOO, but would complain if user wants to
// create a new container named FOO. To work around this, we parse the error
// message to identify failure caused by naming conflict, and try to remove
// the old container FOO.
// See #40443. Sometimes even removal may fail with "no such container" error.
// In that case we have to create the container with a randomized name.
// TODO(random-liu): Remove this work around after docker 1.11 is deprecated.
// TODO(#33189): Monitor the tests to see if the fix is sufficient.
func recoverFromCreationConflictIfNeeded(client libdocker.Interface, createConfig dockertypes.ContainerCreateConfig, err error) (*dockercontainer.ContainerCreateCreatedBody, error) {
	matches := conflictRE.FindStringSubmatch(err.Error())
	if len(matches) != 2 {
		return nil, err
	}

	id := matches[1]
	klog.Warningf("Unable to create pod sandbox due to conflict. Attempting to remove sandbox %q", id)
	rmErr := client.RemoveContainer(id, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
	if rmErr == nil {
		klog.V(2).Infof("Successfully removed conflicting container %q", id)
		return nil, err
	}
	klog.Errorf("Failed to remove the conflicting container %q: %v", id, rmErr)
	// Return if the error is not container not found error.
	if !libdocker.IsContainerNotFoundError(rmErr) {
		return nil, err
	}

	// randomize the name to avoid conflict.
	createConfig.Name = randomizeName(createConfig.Name)
	klog.V(2).Infof("Create the container with randomized name %s", createConfig.Name)
	return client.CreateContainer(createConfig)
}

// transformStartContainerError does regex parsing on returned error
// for where container runtimes are giving less than ideal error messages.
func transformStartContainerError(err error) error {
	if err == nil {
		return nil
	}
	matches := startRE.FindStringSubmatch(err.Error())
	if len(matches) > 0 {
		return fmt.Errorf("executable not found in $PATH")
	}
	return err
}

// ensureSandboxImageExists pulls the sandbox image when it's not present.
func ensureSandboxImageExists(client libdocker.Interface, image string) error {
	_, err := client.InspectImageByRef(image)
	if err == nil {
		return nil
	}
	if !libdocker.IsImageNotFoundError(err) {
		return fmt.Errorf("failed to inspect sandbox image %q: %v", image, err)
	}

	repoToPull, _, _, err := parsers.ParseImageName(image)
	if err != nil {
		return err
	}

	keyring := credentialprovider.NewDockerKeyring()
	creds, withCredentials := keyring.Lookup(repoToPull)
	if !withCredentials {
		klog.V(3).Infof("Pulling image %q without credentials", image)

		err := client.PullImage(image, dockertypes.AuthConfig{}, dockertypes.ImagePullOptions{})
		if err != nil {
			return fmt.Errorf("failed pulling image %q: %v", image, err)
		}

		return nil
	}

	var pullErrs []error
	for _, currentCreds := range creds {
		authConfig := dockertypes.AuthConfig(currentCreds)
		err := client.PullImage(image, authConfig, dockertypes.ImagePullOptions{})
		// If there was no error, return success
		if err == nil {
			return nil
		}

		pullErrs = append(pullErrs, err)
	}

	return utilerrors.NewAggregate(pullErrs)
}

func getAppArmorOpts(profile string) ([]dockerOpt, error) {
	if profile == "" || profile == v1.AppArmorBetaProfileRuntimeDefault {
		// The docker applies the default profile by default.
		return nil, nil
	}

	// Return unconfined profile explicitly
	if profile == v1.AppArmorBetaProfileNameUnconfined {
		return []dockerOpt{{"apparmor", v1.AppArmorBetaProfileNameUnconfined, ""}}, nil
	}

	// Assume validation has already happened.
	profileName := strings.TrimPrefix(profile, v1.AppArmorBetaProfileNamePrefix)
	return []dockerOpt{{"apparmor", profileName, ""}}, nil
}

// fmtDockerOpts formats the docker security options using the given separator.
func fmtDockerOpts(opts []dockerOpt, sep rune) []string {
	fmtOpts := make([]string, len(opts))
	for i, opt := range opts {
		fmtOpts[i] = fmt.Sprintf("%s%c%s", opt.key, sep, opt.value)
	}
	return fmtOpts
}

type dockerOpt struct {
	// The key-value pair passed to docker.
	key, value string
	// The alternative value to use in log/event messages.
	msg string
}

// Expose key/value from  dockerOpt.
func (d dockerOpt) GetKV() (string, string) {
	return d.key, d.value
}

// sharedWriteLimiter limits the total output written across one or more streams.
type sharedWriteLimiter struct {
	delegate io.Writer
	limit    *int64
}

func (w sharedWriteLimiter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	limit := atomic.LoadInt64(w.limit)
	if limit <= 0 {
		return 0, errMaximumWrite
	}
	var truncated bool
	if limit < int64(len(p)) {
		p = p[0:limit]
		truncated = true
	}
	n, err := w.delegate.Write(p)
	if n > 0 {
		atomic.AddInt64(w.limit, -1*int64(n))
	}
	if err == nil && truncated {
		err = errMaximumWrite
	}
	return n, err
}

func sharedLimitWriter(w io.Writer, limit *int64) io.Writer {
	if w == nil {
		return nil
	}
	return &sharedWriteLimiter{
		delegate: w,
		limit:    limit,
	}
}

var errMaximumWrite = errors.New("maximum write")
