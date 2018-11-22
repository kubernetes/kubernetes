/*
Copyright 2017 The Kubernetes Authors.

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

package staticpod

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"sort"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// kubeControllerManagerAddressArg represents the address argument of the kube-controller-manager configuration.
	kubeControllerManagerAddressArg = "address"

	// kubeSchedulerAddressArg represents the address argument of the kube-scheduler configuration.
	kubeSchedulerAddressArg = "address"

	// etcdListenClientURLsArg represents the listen-client-urls argument of the etcd configuration.
	etcdListenClientURLsArg = "listen-client-urls"
)

// ComponentPod returns a Pod object from the container and volume specifications
func ComponentPod(container v1.Container, volumes map[string]v1.Volume) v1.Pod {
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        container.Name,
			Namespace:   metav1.NamespaceSystem,
			Annotations: map[string]string{kubetypes.CriticalPodAnnotationKey: ""},
			// The component and tier labels are useful for quickly identifying the control plane Pods when doing a .List()
			// against Pods in the kube-system namespace. Can for example be used together with the WaitForPodsWithLabel function
			Labels: map[string]string{"component": container.Name, "tier": "control-plane"},
		},
		Spec: v1.PodSpec{
			Containers:        []v1.Container{container},
			PriorityClassName: "system-cluster-critical",
			HostNetwork:       true,
			Volumes:           VolumeMapToSlice(volumes),
		},
	}
}

// ComponentResources returns the v1.ResourceRequirements object needed for allocating a specified amount of the CPU
func ComponentResources(cpu string) v1.ResourceRequirements {
	return v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceName(v1.ResourceCPU): resource.MustParse(cpu),
		},
	}
}

// ComponentProbe is a helper function building a ready v1.Probe object from some simple parameters
func ComponentProbe(cfg *kubeadmapi.InitConfiguration, componentName string, port int, path string, scheme v1.URIScheme) *v1.Probe {
	return &v1.Probe{
		Handler: v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				Host:   GetProbeAddress(cfg, componentName),
				Path:   path,
				Port:   intstr.FromInt(port),
				Scheme: scheme,
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
		FailureThreshold:    8,
	}
}

// EtcdProbe is a helper function for building a shell-based, etcdctl v1.Probe object to healthcheck etcd
func EtcdProbe(cfg *kubeadmapi.InitConfiguration, componentName string, port int, certsDir string, CACertName string, CertName string, KeyName string) *v1.Probe {
	tlsFlags := fmt.Sprintf("--cacert=%[1]s/%[2]s --cert=%[1]s/%[3]s --key=%[1]s/%[4]s", certsDir, CACertName, CertName, KeyName)
	// etcd pod is alive if a linearizable get succeeds.
	cmd := fmt.Sprintf("ETCDCTL_API=3 etcdctl --endpoints=https://[%s]:%d %s get foo", GetProbeAddress(cfg, componentName), port, tlsFlags)

	return &v1.Probe{
		Handler: v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"/bin/sh", "-ec", cmd},
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
		FailureThreshold:    8,
	}
}

// NewVolume creates a v1.Volume with a hostPath mount to the specified location
func NewVolume(name, path string, pathType *v1.HostPathType) v1.Volume {
	return v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: path,
				Type: pathType,
			},
		},
	}
}

// NewVolumeMount creates a v1.VolumeMount to the specified location
func NewVolumeMount(name, path string, readOnly bool) v1.VolumeMount {
	return v1.VolumeMount{
		Name:      name,
		MountPath: path,
		ReadOnly:  readOnly,
	}
}

// VolumeMapToSlice returns a slice of volumes from a map's values
func VolumeMapToSlice(volumes map[string]v1.Volume) []v1.Volume {
	v := make([]v1.Volume, 0, len(volumes))

	for _, vol := range volumes {
		v = append(v, vol)
	}

	sort.Slice(v, func(i, j int) bool {
		return strings.Compare(v[i].Name, v[j].Name) == -1
	})

	return v
}

// VolumeMountMapToSlice returns a slice of volumes from a map's values
func VolumeMountMapToSlice(volumeMounts map[string]v1.VolumeMount) []v1.VolumeMount {
	v := make([]v1.VolumeMount, 0, len(volumeMounts))

	for _, volMount := range volumeMounts {
		v = append(v, volMount)
	}

	sort.Slice(v, func(i, j int) bool {
		return strings.Compare(v[i].Name, v[j].Name) == -1
	})

	return v
}

// GetExtraParameters builds a list of flag arguments two string-string maps, one with default, base commands and one with overrides
func GetExtraParameters(overrides map[string]string, defaults map[string]string) []string {
	var command []string
	for k, v := range overrides {
		if len(v) > 0 {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	for k, v := range defaults {
		if _, overrideExists := overrides[k]; !overrideExists {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	return command
}

// WriteStaticPodToDisk writes a static pod file to disk
func WriteStaticPodToDisk(componentName, manifestDir string, pod v1.Pod) error {

	// creates target folder if not already exists
	if err := os.MkdirAll(manifestDir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q", manifestDir)
	}

	// writes the pod to disk
	serialized, err := util.MarshalToYaml(&pod, v1.SchemeGroupVersion)
	if err != nil {
		return errors.Wrapf(err, "failed to marshal manifest for %q to YAML", componentName)
	}

	filename := kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir)

	if err := ioutil.WriteFile(filename, serialized, 0600); err != nil {
		return errors.Wrapf(err, "failed to write static pod manifest file for %q (%q)", componentName, filename)
	}

	return nil
}

// ReadStaticPodFromDisk reads a static pod file from disk
func ReadStaticPodFromDisk(manifestPath string) (*v1.Pod, error) {
	buf, err := ioutil.ReadFile(manifestPath)
	if err != nil {
		return &v1.Pod{}, errors.Wrapf(err, "failed to read manifest for %q", manifestPath)
	}

	obj, err := util.UnmarshalFromYaml(buf, v1.SchemeGroupVersion)
	if err != nil {
		return &v1.Pod{}, errors.Errorf("failed to unmarshal manifest for %q from YAML: %v", manifestPath, err)
	}

	pod := obj.(*v1.Pod)

	return pod, nil
}

// GetProbeAddress returns an IP address or 127.0.0.1 to use for liveness probes
// in static pod manifests.
func GetProbeAddress(cfg *kubeadmapi.InitConfiguration, componentName string) string {
	switch {
	case componentName == kubeadmconstants.KubeAPIServer:
		// In the case of a self-hosted deployment, the initial host on which kubeadm --init is run,
		// will generate a DaemonSet with a nodeSelector such that all nodes with the label
		// node-role.kubernetes.io/master='' will have the API server deployed to it. Since the init
		// is run only once on an initial host, the API advertise address will be invalid for any
		// future hosts that do not have the same address. Furthermore, since liveness and readiness
		// probes do not support the Downward API we cannot dynamically set the advertise address to
		// the node's IP. The only option then is to use localhost.
		if cfg.LocalAPIEndpoint.AdvertiseAddress != "" {
			return cfg.LocalAPIEndpoint.AdvertiseAddress
		}
	case componentName == kubeadmconstants.KubeControllerManager:
		if addr, exists := cfg.ControllerManager.ExtraArgs[kubeControllerManagerAddressArg]; exists {
			return addr
		}
	case componentName == kubeadmconstants.KubeScheduler:
		if addr, exists := cfg.Scheduler.ExtraArgs[kubeSchedulerAddressArg]; exists {
			return addr
		}
	case componentName == kubeadmconstants.Etcd:
		if cfg.Etcd.Local != nil && cfg.Etcd.Local.ExtraArgs != nil {
			if arg, exists := cfg.Etcd.Local.ExtraArgs[etcdListenClientURLsArg]; exists {
				// Use the first url in the listen-client-urls if multiple url's are specified.
				if strings.ContainsAny(arg, ",") {
					arg = strings.Split(arg, ",")[0]
				}
				parsedURL, err := url.Parse(arg)
				if err != nil || parsedURL.Hostname() == "" {
					break
				}
				// Return the IP if the URL contains an address instead of a name.
				if ip := net.ParseIP(parsedURL.Hostname()); ip != nil {
					// etcdctl doesn't support auto-converting zero addresses into loopback addresses
					if ip.Equal(net.IPv4zero) {
						return "127.0.0.1"
					}
					if ip.Equal(net.IPv6zero) {
						return net.IPv6loopback.String()
					}
					return ip.String()
				}
				// Use the local resolver to try resolving the name within the URL.
				// If the name can not be resolved, return an IPv4 loopback address.
				// Otherwise, select the first valid IPv4 address.
				// If the name does not resolve to an IPv4 address, select the first valid IPv6 address.
				addrs, err := net.LookupIP(parsedURL.Hostname())
				if err != nil {
					break
				}
				var ip net.IP
				for _, addr := range addrs {
					if addr.To4() != nil {
						ip = addr
						break
					}
					if addr.To16() != nil && ip == nil {
						ip = addr
					}
				}
				return ip.String()
			}
		}
	}
	return "127.0.0.1"
}

// ManifestFilesAreEqual compares 2 files. It returns true if their contents are equal, false otherwise
func ManifestFilesAreEqual(path1, path2 string) (bool, error) {
	content1, err := ioutil.ReadFile(path1)
	if err != nil {
		return false, err
	}
	content2, err := ioutil.ReadFile(path2)
	if err != nil {
		return false, err
	}

	return bytes.Equal(content1, content2), nil
}
