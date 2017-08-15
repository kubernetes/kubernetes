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
	"fmt"
	"io/ioutil"
	"os"

	"github.com/ghodss/yaml"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// ComponentPod returns a Pod object from the container and volume specifications
func ComponentPod(container v1.Container, volumes []v1.Volume) v1.Pod {
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        container.Name,
			Namespace:   metav1.NamespaceSystem,
			Annotations: map[string]string{kubetypes.CriticalPodAnnotationKey: ""},
		},
		Spec: v1.PodSpec{
			Containers:  []v1.Container{container},
			HostNetwork: true,
			Volumes:     volumes,
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
func ComponentProbe(port int, path string, scheme v1.URIScheme) *v1.Probe {
	return &v1.Probe{
		Handler: v1.Handler{
			HTTPGet: &v1.HTTPGetAction{
				// Host has to be set to "127.0.0.1" here due to that our static Pods are on the host's network
				Host:   "127.0.0.1",
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

// NewVolume creates a v1.Volume with a hostPath mount to the specified location
func NewVolume(name, path string) v1.Volume {
	return v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: path},
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
		return fmt.Errorf("failed to create directory %q: %v", manifestDir, err)
	}

	// writes the pod to disk
	serialized, err := yaml.Marshal(pod)
	if err != nil {
		return fmt.Errorf("failed to marshal manifest for %q to YAML: %v", componentName, err)
	}

	filename := kubeadmconstants.GetStaticPodFilepath(componentName, manifestDir)

	if err := ioutil.WriteFile(filename, serialized, 0700); err != nil {
		return fmt.Errorf("failed to write static pod manifest file for %q (%q): %v", componentName, filename, err)
	}

	return nil
}
