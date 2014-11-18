/*
Copyright 2014 Google Inc. All rights reserved.

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

package controller

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

// PodController is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodController interface {
	// createPod creates new replicated pods according to the spec.
	createPod(namespace string, template *api.PodTemplateSpec)
	// deletePod deletes the pod identified by podID.
	deletePod(namespace string, podID string) error
}

// RealPodControl is the default implementation of PodController.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) createPod(namespace string, template *api.PodTemplateSpec) {
	desiredLabels := make(labels.Set)
	for k, v := range template.Labels {
		desiredLabels[k] = v
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: desiredLabels,
		},
	}
	if err := api.Scheme.Convert(&template.Spec, &pod.DesiredState.Manifest); err != nil {
		glog.Errorf("Unable to convert pod template: %v", err)
		return
	}
	if labels.Set(pod.Labels).AsSelector().Empty() {
		glog.Errorf("Unable to create pod: no labels")
		return
	}
	if _, err := r.kubeClient.Pods(namespace).Create(pod); err != nil {
		glog.Errorf("Unable to create pod: %v", err)
	}
}

func (r RealPodControl) deletePod(namespace, podID string) error {
	return r.kubeClient.Pods(namespace).Delete(podID)
}
