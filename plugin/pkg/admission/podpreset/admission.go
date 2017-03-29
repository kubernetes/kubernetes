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

package admission

import (
	"fmt"
	"io"
	"reflect"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	settingslisters "k8s.io/kubernetes/pkg/client/listers/settings/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	annotationPrefix = "podpreset.admission.kubernetes.io"
	pluginName       = "PodPreset"
)

func init() {
	admission.RegisterPlugin(pluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// podPresetPlugin is an implementation of admission.Interface.
type podPresetPlugin struct {
	*admission.Handler
	client internalclientset.Interface

	lister settingslisters.PodPresetLister
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&podPresetPlugin{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&podPresetPlugin{})

// NewPlugin creates a new pod preset admission plugin.
func NewPlugin() *podPresetPlugin {
	return &podPresetPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

func (plugin *podPresetPlugin) Validate() error {
	if plugin.client == nil {
		return fmt.Errorf("%s requires a client", pluginName)
	}
	if plugin.lister == nil {
		return fmt.Errorf("%s requires a lister", pluginName)
	}
	return nil
}

func (a *podPresetPlugin) SetInternalKubeClientSet(client internalclientset.Interface) {
	a.client = client
}

func (a *podPresetPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	podPresetInformer := f.Settings().InternalVersion().PodPresets()
	a.lister = podPresetInformer.Lister()
	a.SetReadyFunc(podPresetInformer.Informer().HasSynced)
}

// Admit injects a pod with the specific fields for each pod preset it matches.
func (c *podPresetPlugin) Admit(a admission.Attributes) error {
	// Ignore all calls to subresources or resources other than pods.
	// Ignore all operations other than CREATE.
	if len(a.GetSubresource()) != 0 || a.GetResource().GroupResource() != api.Resource("pods") || a.GetOperation() != admission.Create {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	list, err := c.lister.PodPresets(pod.GetNamespace()).List(labels.Everything())
	if err != nil {
		return fmt.Errorf("listing pod presets failed: %v", err)
	}

	// get the pod presets and iterate over them
	for _, pip := range list {
		selector, err := metav1.LabelSelectorAsSelector(&pip.Spec.Selector)
		if err != nil {
			return fmt.Errorf("listing pod presets for namespace:%s failed: %v", pod.GetNamespace(), err)
		}

		// check if the pod labels match the selector
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}

		glog.V(4).Infof("PodPreset %s matches pod %s labels", pip.GetName(), pod.GetName())

		// merge in policy for Env
		if pip.Spec.Env != nil {
			for i, ctr := range pod.Spec.Containers {
				r, err := mergeEnv(pip, ctr.Env)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())

					return nil
				}
				pod.Spec.Containers[i].Env = r
			}
		}

		// merge in policy for EnvFrom
		if pip.Spec.EnvFrom != nil {
			for i, ctr := range pod.Spec.Containers {
				r, err := mergeEnvFrom(pip, ctr.EnvFrom)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())

					return nil
				}
				pod.Spec.Containers[i].EnvFrom = r
			}
		}

		// merge in policy for VolumeMounts
		if pip.Spec.VolumeMounts != nil {
			for i, ctr := range pod.Spec.Containers {
				r, err := mergeVolumeMounts(pip, ctr.VolumeMounts)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())

					return nil
				}
				pod.Spec.Containers[i].VolumeMounts = r
			}
		}

		// merge in policy for Volumes
		if pip.Spec.Volumes != nil {
			r, err := mergeVolumes(pip, pod.Spec.Volumes)
			if err != nil {
				// add event to pod
				c.addEvent(pod, pip, err.Error())

				return nil
			}
			pod.Spec.Volumes = r
		}

		glog.V(4).Infof("PodPreset %s merged with pod %s successfully", pip.GetName(), pod.GetName())

		// add annotation
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = map[string]string{}
		}
		pod.ObjectMeta.Annotations[fmt.Sprintf("%s/%s", annotationPrefix, pip.GetName())] = pip.GetResourceVersion()
	}

	return nil
}

func mergeEnv(pip *settings.PodPreset, original []api.EnvVar) ([]api.EnvVar, error) {
	// if there were no original envvar just return the pip envvar
	if original == nil {
		return pip.Spec.Env, nil
	}

	orig := map[string]interface{}{}
	for _, v := range original {
		orig[v.Name] = v
	}

	// check for conflicts.
	for _, v := range pip.Spec.Env {
		found, ok := orig[v.Name]
		if !ok {
			// if we don't already have it append it and continue
			original = append(original, v)
			continue
		}

		// make sure they are identical or throw an error
		if !reflect.DeepEqual(found, v) {
			return nil, fmt.Errorf("merging env for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\n in container", pip.GetName(), v.Name, v, found)
		}
	}

	return original, nil
}

func mergeEnvFrom(pip *settings.PodPreset, original []api.EnvFromSource) ([]api.EnvFromSource, error) {
	// if there were no original envfrom just return the pip envfrom
	if original == nil {
		return pip.Spec.EnvFrom, nil
	}

	return append(original, pip.Spec.EnvFrom...), nil
}

func mergeVolumeMounts(pip *settings.PodPreset, original []api.VolumeMount) ([]api.VolumeMount, error) {
	// if there were no original volume mount just return the pip volume mount
	if original == nil {
		return pip.Spec.VolumeMounts, nil
	}

	// first key by name
	orig := map[string]interface{}{}
	for _, v := range original {
		orig[v.Name] = v
	}

	// check for conflicts.
	for _, v := range pip.Spec.VolumeMounts {
		found, ok := orig[v.Name]
		if !ok {
			// if we don't already have it continue
			continue
		}

		// make sure they are identical or throw an error
		if !reflect.DeepEqual(found, v) {
			return nil, fmt.Errorf("merging volume mounts for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\n in container", pip.GetName(), v.Name, v, found)
		}
	}

	// key by mount path
	orig = map[string]interface{}{}
	for _, v := range original {
		orig[v.MountPath] = v
	}

	// check for conflicts.
	for _, v := range pip.Spec.VolumeMounts {
		found, ok := orig[v.MountPath]
		if !ok {
			// if we don't already have it append it and continue
			original = append(original, v)
			continue
		}

		// make sure they are identical or throw an error
		if !reflect.DeepEqual(found, v) {
			return nil, fmt.Errorf("merging volume mounts for %s has a conflict on mount path %s: \n%#v\ndoes not match\n%#v\n in container", pip.GetName(), v.MountPath, v, found)
		}
	}

	return original, nil
}

func mergeVolumes(pip *settings.PodPreset, original []api.Volume) ([]api.Volume, error) {
	// if there were no original volumes just return the pip volumes
	if original == nil {
		return pip.Spec.Volumes, nil
	}

	orig := map[string]api.Volume{}
	for _, v := range original {
		orig[v.Name] = v
	}

	// check for conflicts.
	for _, v := range pip.Spec.Volumes {
		found, ok := orig[v.Name]
		if !ok {
			// if we don't already have it append it and continue
			original = append(original, v)
			continue
		}

		if !reflect.DeepEqual(found, v) {
			return nil, fmt.Errorf("merging volumes for %s has a conflict on %s: \n%#v\ndoes not match\n%#v\nin pod spec", pip.GetName(), v.Name, v, found)
		}
	}

	return original, nil
}

func (c *podPresetPlugin) addEvent(pod *api.Pod, pip *settings.PodPreset, message string) {
	ref, err := api.GetReference(api.Scheme, pod)
	if err != nil {
		glog.Errorf("pip %s: get reference for pod %s failed: %v", pip.GetName(), pod.GetName(), err)
		return
	}

	e := &api.Event{
		InvolvedObject: *ref,
		Message:        message,
		Source: api.EventSource{
			Component: fmt.Sprintf("pip %s", pip.GetName()),
		},
		Type: "Warning",
	}

	if _, err := c.client.Core().Events(pod.GetNamespace()).Create(e); err != nil {
		glog.Errorf("pip %s: creating pod event failed: %v", pip.GetName(), err)
		return
	}
}
