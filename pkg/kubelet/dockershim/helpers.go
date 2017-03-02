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
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/blang/semver"
	dockertypes "github.com/docker/engine-api/types"
	dockerfilters "github.com/docker/engine-api/types/filters"
	dockernat "github.com/docker/go-connections/nat"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	annotationPrefix = "annotation."
)

var (
	conflictRE = regexp.MustCompile(`Conflict. (?:.)+ is already in use by container ([0-9a-z]+)`)

	// Docker changes the security option separator from ':' to '=' in the 1.23
	// API version.
	optsSeparatorChangeVersion = semver.MustParse(dockertools.SecurityOptSeparatorChangeVersion)
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
// Each element in the string is in the form of:
// '<HostPath>:<ContainerPath>', or
// '<HostPath>:<ContainerPath>:ro', if the path is read only, or
// '<HostPath>:<ContainerPath>:Z', if the volume requires SELinux
// relabeling and the pod provides an SELinux label
func generateMountBindings(mounts []*runtimeapi.Mount) (result []string) {
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.HostPath, m.ContainerPath)
		readOnly := m.Readonly
		if readOnly {
			bind += ":ro"
		}
		// Only request relabeling if the pod provides an SELinux context. If the pod
		// does not provide an SELinux context relabeling will label the volume with
		// the container's randomly allocated MCS label. This would restrict access
		// to the volume to the container which mounts it first.
		if m.SelinuxRelabel {
			if readOnly {
				bind += ",Z"
			} else {
				bind += ":Z"
			}
		}
		result = append(result, bind)
	}
	return
}

func makePortsAndBindings(pm []*runtimeapi.PortMapping) (map[dockernat.Port]struct{}, map[dockernat.Port][]dockernat.PortBinding) {
	exposedPorts := map[dockernat.Port]struct{}{}
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
		switch strings.ToUpper(string(port.Protocol)) {
		case "UDP":
			protocol = "/udp"
		case "TCP":
			protocol = "/tcp"
		default:
			glog.Warningf("Unknown protocol %q: defaulting to TCP", port.Protocol)
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

// getContainerSecurityOpt gets container security options from container and sandbox config, currently from sandbox
// annotations.
// It is an experimental feature and may be promoted to official runtime api in the future.
func getContainerSecurityOpts(containerName string, sandboxConfig *runtimeapi.PodSandboxConfig, seccompProfileRoot string, separator rune) ([]string, error) {
	appArmorOpts, err := dockertools.GetAppArmorOpts(sandboxConfig.GetAnnotations(), containerName)
	if err != nil {
		return nil, err
	}
	seccompOpts, err := dockertools.GetSeccompOpts(sandboxConfig.GetAnnotations(), containerName, seccompProfileRoot)
	if err != nil {
		return nil, err
	}
	securityOpts := append(appArmorOpts, seccompOpts...)
	fmtOpts := dockertools.FmtDockerOpts(securityOpts, separator)
	return fmtOpts, nil
}

func getSandboxSecurityOpts(sandboxConfig *runtimeapi.PodSandboxConfig, seccompProfileRoot string, separator rune) ([]string, error) {
	// sandboxContainerName doesn't exist in the pod, so pod security options will be returned by default.
	return getContainerSecurityOpts(sandboxContainerName, sandboxConfig, seccompProfileRoot, separator)
}

func getNetworkNamespace(c *dockertypes.ContainerJSON) string {
	if c.State.Pid == 0 {
		// Docker reports pid 0 for an exited container. We can't use it to
		// check the network namespace, so return an empty string instead.
		glog.V(4).Infof("Cannot find network namespace for the terminated container %q", c.ID)
		return ""
	}
	return fmt.Sprintf(dockerNetNSFmt, c.State.Pid)
}

// getSysctlsFromAnnotations gets sysctls from annotations.
func getSysctlsFromAnnotations(annotations map[string]string) (map[string]string, error) {
	var results map[string]string

	sysctls, unsafeSysctls, err := v1.SysctlsFromPodAnnotations(annotations)
	if err != nil {
		return nil, err
	}
	if len(sysctls)+len(unsafeSysctls) > 0 {
		results = make(map[string]string, len(sysctls)+len(unsafeSysctls))
		for _, c := range sysctls {
			results[c.Name] = c.Value
		}
		for _, c := range unsafeSysctls {
			results[c.Name] = c.Value
		}
	}

	return results, nil
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

// getUserFromImageUser gets uid or user name of the image user.
// If user is numeric, it will be treated as uid; or else, it is treated as user name.
func getUserFromImageUser(imageUser string) (*int64, string) {
	user := dockertools.GetUserFromImageUser(imageUser)
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
// failed due to "device or resource busy", it is possbile that docker did
// not clean up properly and has inconsistent internal state. Docker would
// not report the existence of FOO, but would complain if user wants to
// create a new container named FOO. To work around this, we parse the error
// message to identify failure caused by naming conflict, and try to remove
// the old container FOO.
// See #40443. Sometimes even removal may fail with "no such container" error.
// In that case we have to create the container with a randomized name.
// TODO(random-liu): Remove this work around after docker 1.11 is deprecated.
// TODO(#33189): Monitor the tests to see if the fix is sufficent.
func recoverFromCreationConflictIfNeeded(client dockertools.DockerInterface, createConfig dockertypes.ContainerCreateConfig, err error) (*dockertypes.ContainerCreateResponse, error) {
	matches := conflictRE.FindStringSubmatch(err.Error())
	if len(matches) != 2 {
		return nil, err
	}

	id := matches[1]
	glog.Warningf("Unable to create pod sandbox due to conflict. Attempting to remove sandbox %q", id)
	if rmErr := client.RemoveContainer(id, dockertypes.ContainerRemoveOptions{RemoveVolumes: true}); rmErr == nil {
		glog.V(2).Infof("Successfully removed conflicting container %q", id)
		return nil, err
	} else {
		glog.Errorf("Failed to remove the conflicting container %q: %v", id, rmErr)
		// Return if the error is not container not found error.
		if !dockertools.IsContainerNotFoundError(rmErr) {
			return nil, err
		}
	}

	// randomize the name to avoid conflict.
	createConfig.Name = randomizeName(createConfig.Name)
	glog.V(2).Infof("Create the container with randomized name %s", createConfig.Name)
	return client.CreateContainer(createConfig)
}

// getSecurityOptSeparator returns the security option separator based on the
// docker API version.
// TODO: Remove this function along with the relevant code when we no longer
// need to support docker 1.10.
func getSecurityOptSeparator(v *semver.Version) rune {
	switch v.Compare(optsSeparatorChangeVersion) {
	case -1:
		// Current version is less than the API change version; use the old
		// separator.
		return dockertools.SecurityOptSeparatorOld
	default:
		return dockertools.SecurityOptSeparatorNew
	}
}

// ensureSandboxImageExists pulls the sandbox image when it's not present.
func ensureSandboxImageExists(client dockertools.DockerInterface, image string) error {
	_, err := client.InspectImageByRef(image)
	if err == nil {
		return nil
	}
	if !dockertools.IsImageNotFoundError(err) {
		return fmt.Errorf("failed to inspect sandbox image %q: %v", image, err)
	}
	err = client.PullImage(image, dockertypes.AuthConfig{}, dockertypes.ImagePullOptions{})
	if err != nil {
		return fmt.Errorf("unable to pull sandbox image %q: %v", image, err)
	}
	return nil
}
