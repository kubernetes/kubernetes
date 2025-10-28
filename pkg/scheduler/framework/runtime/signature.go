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

type podSignatureMakerImpl struct {
	elements Elements
	signable bool
}

func newPodSignatureMaker() *podSignatureMakerImpl {
	return &podSignatureMakerImpl{
		elements: Elements{
			Pod:    map[string]string{},
			Plugin: map[string]string{},
		},
		signable: true,
	}
}

// The given pod cannot be signed by the given plugin.
func (s *podSignatureMakerImpl) Unsignable() {
	s.signable = false
}

// Add a part of the pod (spec, etc) to the signature if it hasn't been already.
// The pod path should be in dot notation (so pod.Spec.NodeName should be "Spec.NodeName")
// to avoid collisions.
func (s *podSignatureMakerImpl) AddPodElement(podPath string, object any) error {
	if _, found := s.elements.Pod[podPath]; !found {
		return s.addElement(s.elements.Pod, podPath, object)
	}
	return nil
}

// Add a plugin specific element to the signature. The name should be the plugin name to
// avoid collisions.
func (s *podSignatureMakerImpl) AddPluginElement(pluginName string, object any) error {
	return s.addElement(s.elements.Plugin, pluginName, object)
}

// Add an element to the signature based on a key and an object to use as a value.
// This assumes the object is json serializable.
// Note that the golang json serializer has fixed ordering for structs and maps, so it is stable:
//
//	https://stackoverflow.com/questions/18668652/how-to-produce-json-with-sorted-keys-in-go
func (s *podSignatureMakerImpl) addElement(elemMap map[string]string, elementName string, object any) error {
	marshalled, err := json.Marshal(object)
	if err != nil {
		return err
	}
	elemMap[elementName] = string(marshalled)
	return nil
}

// Marshal the signature into a string.
func (s *podSignatureMakerImpl) Marshal() ([]byte, error) {
	if s.signable {
		return json.Marshal(s.elements)
	} else {
		return []byte(fwk.Unsignable), nil
	}
}

// Add signature components that are not plugin specific.
func (s *podSignatureMakerImpl) AddNonPluginElements(pod *v1.Pod) error {
	return s.AddPodElement("Spec.SchedulerName", pod.Spec.SchedulerName)
}

// Common signature element: the pod's Volumes.  Note that
// we exclude ConfigMap and Secret volumes because they are synthetic.
func (s *podSignatureMakerImpl) AddSignatureVolumes(pod *v1.Pod) error {
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
