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
	"strconv"
	"strings"

	dockertypes "github.com/docker/engine-api/types"
	dockerfilters "github.com/docker/engine-api/types/filters"
	dockerapiversion "github.com/docker/engine-api/types/versions"
	dockernat "github.com/docker/go-connections/nat"
	"github.com/golang/glog"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// apiVersion implements kubecontainer.Version interface by implementing
// Compare() and String(). It uses the compare function of engine-api to
// compare docker apiversions.
type apiVersion string

func (v apiVersion) String() string {
	return string(v)
}

func (v apiVersion) Compare(other string) (int, error) {
	if dockerapiversion.LessThan(string(v), other) {
		return -1, nil
	} else if dockerapiversion.GreaterThan(string(v), other) {
		return 1, nil
	}
	return 0, nil
}

// generateEnvList converts KeyValue list to a list of strings, in the form of
// '<key>=<value>', which can be understood by docker.
func generateEnvList(envs []*runtimeApi.KeyValue) (result []string) {
	for _, env := range envs {
		result = append(result, fmt.Sprintf("%s=%s", env.GetKey(), env.GetValue()))
	}
	return
}

// Merge annotations and labels because docker supports only labels.
// TODO: Need to be able to distinguish annotations from labels; otherwise, we
// couldn't restore the information when reading the labels back from docker.
func makeLabels(labels, annotations map[string]string) map[string]string {
	merged := make(map[string]string)
	for k, v := range labels {
		merged[k] = v
	}
	for k, v := range annotations {
		if _, ok := merged[k]; !ok {
			// Don't overwrite the key if it already exists.
			merged[k] = v
		}
	}
	return merged
}

// generateMountBindings converts the mount list to a list of strings that
// can be understood by docker.
// Each element in the string is in the form of:
// '<HostPath>:<ContainerPath>', or
// '<HostPath>:<ContainerPath>:ro', if the path is read only, or
// '<HostPath>:<ContainerPath>:Z', if the volume requires SELinux
// relabeling and the pod provides an SELinux label
func generateMountBindings(mounts []*runtimeApi.Mount) (result []string) {
	// TODO: resolve podHasSELinuxLabel
	for _, m := range mounts {
		bind := fmt.Sprintf("%s:%s", m.GetHostPath(), m.GetContainerPath())
		readOnly := m.GetReadonly()
		if readOnly {
			bind += ":ro"
		}
		if m.GetSelinuxRelabel() {
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

func makePortsAndBindings(pm []*runtimeApi.PortMapping) (map[dockernat.Port]struct{}, map[dockernat.Port][]dockernat.PortBinding) {
	exposedPorts := map[dockernat.Port]struct{}{}
	portBindings := map[dockernat.Port][]dockernat.PortBinding{}
	for _, port := range pm {
		exteriorPort := port.GetHostPort()
		if exteriorPort == 0 {
			// No need to do port binding when HostPort is not specified
			continue
		}
		interiorPort := port.GetContainerPort()
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		var protocol string
		switch strings.ToUpper(string(port.GetProtocol())) {
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
			HostIP:   port.GetHostIp(),
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

// TODO: Seccomp support. Need to figure out how to pass seccomp options
// through the runtime API (annotations?).See dockerManager.getSecurityOpts()
// for the details. Always set the default seccomp profile for now.
// Also need to support syntax for different docker versions.
func getSeccompOpts() string {
	return fmt.Sprintf("%s=%s", "seccomp", defaultSeccompProfile)
}

func getNetworkNamespace(c *dockertypes.ContainerJSON) string {
	return fmt.Sprintf(dockerNetNSFmt, c.State.Pid)
}

// dockerFilter wraps around dockerfilters.Args and provides methods to modify
// the filter easily.
type dockerFilter struct {
	f *dockerfilters.Args
}

func newDockerFilter(args *dockerfilters.Args) *dockerFilter {
	return &dockerFilter{f: args}
}

func (f *dockerFilter) Add(key, value string) {
	f.Add(key, value)
}

func (f *dockerFilter) AddLabel(key, value string) {
	f.Add("label", fmt.Sprintf("%s=%s", key, value))
}
