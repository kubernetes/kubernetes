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
	"io"
	"math"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/patches"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/users"
)

const (
	// kubeControllerManagerBindAddressArg represents the bind-address argument of the kube-controller-manager configuration.
	kubeControllerManagerBindAddressArg = "bind-address"

	// kubeSchedulerBindAddressArg represents the bind-address argument of the kube-scheduler configuration.
	kubeSchedulerBindAddressArg = "bind-address"
)

var (
	usersAndGroups     *users.UsersAndGroups
	usersAndGroupsOnce sync.Once
)

// ComponentPod returns a Pod object from the container, volume and annotations specifications
func ComponentPod(container v1.Container, volumes map[string]v1.Volume, annotations map[string]string) v1.Pod {
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      container.Name,
			Namespace: metav1.NamespaceSystem,
			// The component and tier labels are useful for quickly identifying the control plane Pods when doing a .List()
			// against Pods in the kube-system namespace. Can for example be used together with the WaitForPodsWithLabel function
			Labels:      map[string]string{"component": container.Name, "tier": kubeadmconstants.ControlPlaneTier},
			Annotations: annotations,
		},
		Spec: v1.PodSpec{
			Containers:        []v1.Container{container},
			PriorityClassName: "system-node-critical",
			HostNetwork:       true,
			Volumes:           VolumeMapToSlice(volumes),
			SecurityContext: &v1.PodSecurityContext{
				SeccompProfile: &v1.SeccompProfile{
					Type: v1.SeccompProfileTypeRuntimeDefault,
				},
			},
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

// PatchStaticPod applies patches stored in patchesDir to a static Pod.
func PatchStaticPod(pod *v1.Pod, patchesDir string, output io.Writer) (*v1.Pod, error) {
	// Marshal the Pod manifest into YAML.
	podYAML, err := kubeadmutil.MarshalToYaml(pod, v1.SchemeGroupVersion)
	if err != nil {
		return pod, errors.Wrapf(err, "failed to marshal Pod manifest to YAML")
	}

	patchManager, err := patches.GetPatchManagerForPath(patchesDir, patches.KnownTargets(), output)
	if err != nil {
		return pod, err
	}

	patchTarget := &patches.PatchTarget{
		Name:                      pod.Name,
		StrategicMergePatchObject: v1.Pod{},
		Data:                      podYAML,
	}
	if err := patchManager.ApplyPatchesToTarget(patchTarget); err != nil {
		return pod, err
	}

	obj, err := kubeadmutil.UnmarshalFromYaml(patchTarget.Data, v1.SchemeGroupVersion)
	if err != nil {
		return pod, errors.Wrap(err, "failed to unmarshal patched manifest from YAML")
	}

	pod2, ok := obj.(*v1.Pod)
	if !ok {
		return pod, errors.Wrap(err, "patched manifest is not a valid Pod object")
	}

	return pod2, nil
}

// WriteStaticPodToDisk writes a static pod file to disk
func WriteStaticPodToDisk(componentName, manifestDir string, pod v1.Pod) error {

	// creates target folder if not already exists
	if err := os.MkdirAll(manifestDir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q", manifestDir)
	}

	// writes the pod to disk
	serialized, err := kubeadmutil.MarshalToYaml(&pod, v1.SchemeGroupVersion)
	if err != nil {
		return errors.Wrapf(err, "failed to marshal manifest for %q to YAML", componentName)
	}

	filename := kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir)

	if err := os.WriteFile(filename, serialized, 0600); err != nil {
		return errors.Wrapf(err, "failed to write static pod manifest file for %q (%q)", componentName, filename)
	}

	return nil
}

// ReadStaticPodFromDisk reads a static pod file from disk
func ReadStaticPodFromDisk(manifestPath string) (*v1.Pod, error) {
	buf, err := os.ReadFile(manifestPath)
	if err != nil {
		return &v1.Pod{}, errors.Wrapf(err, "failed to read manifest for %q", manifestPath)
	}

	obj, err := kubeadmutil.UnmarshalFromYaml(buf, v1.SchemeGroupVersion)
	if err != nil {
		return &v1.Pod{}, errors.Errorf("failed to unmarshal manifest for %q from YAML: %v", manifestPath, err)
	}

	pod := obj.(*v1.Pod)

	return pod, nil
}

// LivenessProbe creates a Probe object with a HTTPGet handler
func LivenessProbe(host, path string, port int, scheme v1.URIScheme) *v1.Probe {
	// sets initialDelaySeconds same as periodSeconds to skip one period before running a check
	return createHTTPProbe(host, path, port, scheme, 10, 15, 8, 10)
}

// ReadinessProbe creates a Probe object with a HTTPGet handler
func ReadinessProbe(host, path string, port int, scheme v1.URIScheme) *v1.Probe {
	// sets initialDelaySeconds as '0' because we don't want to delay user infrastructure checks
	// looking for "ready" status on kubeadm static Pods
	return createHTTPProbe(host, path, port, scheme, 0, 15, 3, 1)
}

// StartupProbe creates a Probe object with a HTTPGet handler
func StartupProbe(host, path string, port int, scheme v1.URIScheme, timeoutForControlPlane *metav1.Duration) *v1.Probe {
	periodSeconds, timeoutForControlPlaneSeconds := int32(10), kubeadmconstants.DefaultControlPlaneTimeout.Seconds()
	if timeoutForControlPlane != nil {
		timeoutForControlPlaneSeconds = timeoutForControlPlane.Seconds()
	}
	// sets failureThreshold big enough to guarantee the full timeout can cover the worst case scenario for the control-plane to come alive
	// we ignore initialDelaySeconds in the calculation here for simplicity
	failureThreshold := int32(math.Ceil(timeoutForControlPlaneSeconds / float64(periodSeconds)))
	// sets initialDelaySeconds same as periodSeconds to skip one period before running a check
	return createHTTPProbe(host, path, port, scheme, periodSeconds, 15, failureThreshold, periodSeconds)
}

func createHTTPProbe(host, path string, port int, scheme v1.URIScheme, initialDelaySeconds, timeoutSeconds, failureThreshold, periodSeconds int32) *v1.Probe {
	return &v1.Probe{
		ProbeHandler: v1.ProbeHandler{
			HTTPGet: &v1.HTTPGetAction{
				Host:   host,
				Path:   path,
				Port:   intstr.FromInt(port),
				Scheme: scheme,
			},
		},
		InitialDelaySeconds: initialDelaySeconds,
		TimeoutSeconds:      timeoutSeconds,
		FailureThreshold:    failureThreshold,
		PeriodSeconds:       periodSeconds,
	}
}

// GetAPIServerProbeAddress returns the probe address for the API server
func GetAPIServerProbeAddress(endpoint *kubeadmapi.APIEndpoint) string {
	if endpoint != nil && endpoint.AdvertiseAddress != "" {
		return getProbeAddress(endpoint.AdvertiseAddress)
	}

	return "127.0.0.1"
}

// GetControllerManagerProbeAddress returns the kubernetes controller manager probe address
func GetControllerManagerProbeAddress(cfg *kubeadmapi.ClusterConfiguration) string {
	if addr, exists := cfg.ControllerManager.ExtraArgs[kubeControllerManagerBindAddressArg]; exists {
		return getProbeAddress(addr)
	}
	return "127.0.0.1"
}

// GetSchedulerProbeAddress returns the kubernetes scheduler probe address
func GetSchedulerProbeAddress(cfg *kubeadmapi.ClusterConfiguration) string {
	if addr, exists := cfg.Scheduler.ExtraArgs[kubeSchedulerBindAddressArg]; exists {
		return getProbeAddress(addr)
	}
	return "127.0.0.1"
}

// GetEtcdProbeEndpoint takes a kubeadm Etcd configuration object and attempts to parse
// the first URL in the listen-metrics-urls argument, returning an etcd probe hostname,
// port and scheme
func GetEtcdProbeEndpoint(cfg *kubeadmapi.Etcd, isIPv6 bool) (string, int, v1.URIScheme) {
	localhost := "127.0.0.1"
	if isIPv6 {
		localhost = "::1"
	}
	if cfg.Local == nil || cfg.Local.ExtraArgs == nil {
		return localhost, kubeadmconstants.EtcdMetricsPort, v1.URISchemeHTTP
	}
	if arg, exists := cfg.Local.ExtraArgs["listen-metrics-urls"]; exists {
		// Use the first url in the listen-metrics-urls if multiple URL's are specified.
		arg = strings.Split(arg, ",")[0]
		parsedURL, err := url.Parse(arg)
		if err != nil {
			return localhost, kubeadmconstants.EtcdMetricsPort, v1.URISchemeHTTP
		}
		// Parse scheme
		scheme := v1.URISchemeHTTP
		if parsedURL.Scheme == "https" {
			scheme = v1.URISchemeHTTPS
		}
		// Parse hostname
		hostname := parsedURL.Hostname()
		if len(hostname) == 0 {
			hostname = localhost
		}
		// Parse port
		port := kubeadmconstants.EtcdMetricsPort
		portStr := parsedURL.Port()
		if len(portStr) != 0 {
			p, err := kubeadmutil.ParsePort(portStr)
			if err == nil {
				port = p
			}
		}
		return hostname, port, scheme
	}
	return localhost, kubeadmconstants.EtcdMetricsPort, v1.URISchemeHTTP
}

// ManifestFilesAreEqual compares 2 files. It returns true if their contents are equal, false otherwise
func ManifestFilesAreEqual(path1, path2 string) (bool, error) {
	content1, err := os.ReadFile(path1)
	if err != nil {
		return false, err
	}
	content2, err := os.ReadFile(path2)
	if err != nil {
		return false, err
	}

	return bytes.Equal(content1, content2), nil
}

// getProbeAddress returns a valid probe address.
// Kubeadm uses the bind-address to configure the probe address. It's common to use the
// unspecified address "0.0.0.0" or "::" as bind-address when we want to listen in all interfaces,
// however this address can't be used as probe #86504.
// If the address is an unspecified address getProbeAddress returns empty,
// that means that kubelet will use the PodIP as probe address.
func getProbeAddress(addr string) string {
	if addr == "0.0.0.0" || addr == "::" {
		return ""
	}
	return addr
}

func GetUsersAndGroups() (*users.UsersAndGroups, error) {
	var err error
	usersAndGroupsOnce.Do(func() {
		usersAndGroups, err = users.AddUsersAndGroups()
	})
	return usersAndGroups, err
}
