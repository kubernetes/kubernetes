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
	"encoding/json"
	"fmt"
	"io"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	annotationPrefix = "podinjectionpolicy.admission.kubernetes.io/"
	pluginName       = "PodInjectionPolicy"
)

func init() {
	admission.RegisterPlugin(pluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// podInjectionPolicyPlugin is an implementation of admission.Interface.
type podInjectionPolicyPlugin struct {
	*admission.Handler
	client internalclientset.Interface

	reflector *cache.Reflector
	stopChan  chan struct{}
	store     cache.Store
}

var _ = kubeapiserveradmission.WantsInternalClientSet(&podInjectionPolicyPlugin{})

// NewPlugin creates a new pod injection policy admission plugin.
func NewPlugin() *podInjectionPolicyPlugin {
	return &podInjectionPolicyPlugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

func (plugin *podInjectionPolicyPlugin) Validate() error {
	if plugin.store == nil {
		return fmt.Errorf("%s requires an client", pluginName)
	}
	return nil
}

func (a *podInjectionPolicyPlugin) SetInternalClientSet(client internalclientset.Interface) {
	a.client = client
	a.store = cache.NewStore(cache.MetaNamespaceKeyFunc)
	a.reflector = cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.Apps().PodInjectionPolicies(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Apps().PodInjectionPolicies(v1.NamespaceAll).Watch(options)
			},
		},
		&apps.PodInjectionPolicy{},
		a.store,
		0,
	)
	a.Run()
}

func (a *podInjectionPolicyPlugin) Run() {
	if a.stopChan == nil {
		a.stopChan = make(chan struct{})
	}
	a.reflector.RunUntil(a.stopChan)
}

func (a *podInjectionPolicyPlugin) Stop() {
	if a.stopChan != nil {
		close(a.stopChan)
		a.stopChan = nil
	}
}

// Admit injects a pod with the specific fields for each pod injection policy it matches.
func (c *podInjectionPolicyPlugin) Admit(a admission.Attributes) error {
	// Ignore all calls to subresources or resources other than pods.
	if len(a.GetSubresource()) != 0 || a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return errors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	// get the pod injection policies and iterate over them
	for _, p := range c.store.List() {
		pip, ok := p.(apps.PodInjectionPolicy)
		if !ok {
			return fmt.Errorf("error converting pod injection policy to PodInjectionPolicy: %v", p)
		}

		// make sure the pip is for the same namespace as our pod
		if pod.GetNamespace() != pip.GetNamespace() {
			continue
		}

		selector, err := metav1.LabelSelectorAsSelector(&pip.Spec.Selector)
		if err != nil {
			return fmt.Errorf("listing pod injection policies for namespace:%s failed: %v", pod.GetNamespace(), err)
		}

		// check if the pod labels match the selector
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}

		glog.V(4).Infof("PodInjectionPolicy %s matches pod %s labels", pip.GetName(), pod.GetName())

		// merge in policy for Env
		if pip.Spec.Env != nil {
			mod, err := json.Marshal(pip.Spec.Env)
			if err != nil {
				// add event to pod
				c.addEvent(pod, pip, err.Error())
				return fmt.Errorf("marshal of pip Env failed: %v", err)
			}

			for i, ctr := range pod.Spec.Containers {
				orig, err := json.Marshal(ctr.Env)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("marshal of container Env failed: %v", err)
				}

				result, err := strategicpatch.StrategicMergePatch(orig, mod, []api.EnvVar{})
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("Merge error on pip %s for Env: %v", pip.GetName(), err)
				}

				var r []api.EnvVar
				if err := json.Unmarshal(result, &r); err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("unmarshal of pip Env failed: %v", err)
				}

				pod.Spec.Containers[i].Env = r
			}
		}

		// merge in policy for EnvFrom
		if pip.Spec.EnvFrom != nil {
			mod, err := json.Marshal(pip.Spec.EnvFrom)
			if err != nil {
				// add event to pod
				c.addEvent(pod, pip, err.Error())
				return fmt.Errorf("marshal of pip EnvFrom failed: %v", err)
			}

			for i, ctr := range pod.Spec.Containers {
				orig, err := json.Marshal(ctr.EnvFrom)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("marshal of container EnvFrom failed: %v", err)
				}

				result, err := strategicpatch.StrategicMergePatch(orig, mod, []api.EnvFromSource{})
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("Merge error on pip %s for EnvFrom: %v", pip.GetName(), err)
				}

				var r []api.EnvFromSource
				if err := json.Unmarshal(result, &r); err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("unmarshal of pip EnvFrom failed: %v", err)
				}

				pod.Spec.Containers[i].EnvFrom = r
			}
		}

		// merge in policy for VolumeMounts
		if pip.Spec.VolumeMounts != nil {
			mod, err := json.Marshal(pip.Spec.VolumeMounts)
			if err != nil {
				// add event to pod
				c.addEvent(pod, pip, err.Error())
				return fmt.Errorf("marshal of pip VolumeMounts failed: %v", err)
			}

			for i, ctr := range pod.Spec.Containers {
				orig, err := json.Marshal(ctr.VolumeMounts)
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("marshal of container VolumeMounts failed: %v", err)
				}

				result, err := strategicpatch.StrategicMergePatch(orig, mod, []api.VolumeMount{})
				if err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("Merge error on pip %s for VolumeMounts: %v", pip.GetName(), err)
				}

				var r []api.VolumeMount
				if err := json.Unmarshal(result, &r); err != nil {
					// add event to pod
					c.addEvent(pod, pip, err.Error())
					return fmt.Errorf("unmarshal of pip VolumeMounts failed: %v", err)
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

				return err
			}
			pod.Spec.Volumes = r
		}

		glog.V(4).Infof("PodInjectionPolicy %s merged with pod %s successfully", pip.GetName(), pod.GetName())

		// add annotation
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = map[string]string{}
		}
		pod.ObjectMeta.Annotations[fmt.Sprintf("%s/%s", annotationPrefix, pip.GetName())] = pip.GetResourceVersion()
	}

	return nil
}

func mergeVolumes(pip apps.PodInjectionPolicy, original []api.Volume) ([]api.Volume, error) {
	mod, err := json.Marshal(pip.Spec.Volumes)
	if err != nil {
		return nil, fmt.Errorf("marshal of pip Volumes failed: %v", err)
	}

	orig, err := json.Marshal(original)
	if err != nil {
		return nil, fmt.Errorf("marshal of container Volumes failed: %v", err)
	}

	result, err := strategicpatch.StrategicMergePatch(orig, mod, []api.Volume{})
	if err != nil {
		return nil, fmt.Errorf("Merge error on pip %s for Volumes: %v", pip.GetName(), err)
	}
	var r []api.Volume
	if err := json.Unmarshal(result, &r); err != nil {
		return nil, fmt.Errorf("unmarshal of pip Volumes failed: %v", err)
	}

	return r, nil
}

func (c *podInjectionPolicyPlugin) addEvent(pod *api.Pod, pip apps.PodInjectionPolicy, message string) {
	ref, err := api.GetReference(api.Scheme, pod)
	if err != nil {
		glog.Errorf("pip %s: get reference for pod %s failed: %v", pip.GetName(), pod.GetName(), err)
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
	}
}
