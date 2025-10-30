/*
Copyright 2025 The Kubernetes Authors.

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

package runtime

import (
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

type Elements struct {
	Pod    map[string]string
	Plugin map[string]string
}

type podSignatureBuilderImpl struct {
	elements Elements
	signable bool
}

var _ fwk.PodSignatureBuilder = &podSignatureBuilderImpl{}

func newPodSignatureBuilder() *podSignatureBuilderImpl {
	return &podSignatureBuilderImpl{
		elements: Elements{
			Pod:    map[string]string{},
			Plugin: map[string]string{},
		},
		signable: true,
	}
}

// The given pod cannot be signed by the given plugin.
func (s *podSignatureBuilderImpl) Unsignable() {
	s.signable = false
}

// Add a part of the pod (spec, etc) to the signature if it hasn't been already.
// The pod path should be in dot notation (so pod.Spec.NodeName should be "Spec.NodeName")
// to avoid collisions.
func (s *podSignatureBuilderImpl) AddPodElement(podPath string, object any) error {
	if _, found := s.elements.Pod[podPath]; !found {
		return s.addElement(s.elements.Pod, podPath, object)
	}
	return nil
}

// Add a plugin specific element to the signature. The name should be the plugin name to
// avoid collisions.
func (s *podSignatureBuilderImpl) AddPluginElement(pluginName string, object any) error {
	return s.addElement(s.elements.Plugin, pluginName, object)
}

// Add an element to the signature based on a key and an object to use as a value.
// This assumes the object is json serializable.
// Note that the golang json serializer has fixed ordering for structs and maps, so it is stable:
//
//	https://stackoverflow.com/questions/18668652/how-to-produce-json-with-sorted-keys-in-go
func (s *podSignatureBuilderImpl) addElement(elemMap map[string]string, elementName string, object any) error {
	marshalled, err := json.Marshal(object)
	if err != nil {
		return err
	}
	elemMap[elementName] = string(marshalled)
	return nil
}

// Marshal the signature into a string.
func (s *podSignatureBuilderImpl) Build() (string, error) {
	if s.signable {
		res, err := json.Marshal(s.elements)
		return string(res), err
	} else {
		return fwk.Unsignable, nil
	}
}

// Add signature components that are not plugin specific.
func (s *podSignatureBuilderImpl) AddNonPluginElements(pod *v1.Pod) error {
	return s.AddPodElement("Spec.SchedulerName", pod.Spec.SchedulerName)
}

// Common signature element: the pod's Volumes.  Note that
// we exclude ConfigMap and Secret volumes because they are synthetic.
func (s *podSignatureBuilderImpl) AddSignatureVolumes(pod *v1.Pod) error {
	if _, found := s.elements.Pod["_SignatureVolumes"]; !found {
		volumes := []v1.Volume{}
		for _, volume := range pod.Spec.Volumes {
			if volume.VolumeSource.ConfigMap == nil && volume.VolumeSource.Secret == nil {
				volumes = append(volumes, volume)
			}
		}
		return s.AddPodElement("_SignatureVolumes", volumes)
	}
	return nil
}
