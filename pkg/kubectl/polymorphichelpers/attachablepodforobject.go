/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"fmt"
	"sort"
	"time"

	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	apiv1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
)

// attachablePodForObject returns the pod to which to attach given an object.
func attachablePodForObject(restClientGetter genericclioptions.RESTClientGetter, object runtime.Object, timeout time.Duration) (*api.Pod, error) {
	switch t := object.(type) {
	case *api.Pod:
		return t, nil
	case *corev1.Pod:
		internalPod := &api.Pod{}
		err := apiv1.Convert_v1_Pod_To_core_Pod(t, internalPod, nil)
		return internalPod, err

	}

	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	clientset, err := internalclientset.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}

	namespace, selector, err := SelectorsForObject(object)
	if err != nil {
		return nil, fmt.Errorf("cannot attach to %T: %v", object, err)
	}
	sortBy := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
	pod, _, err := GetFirstPod(clientset.Core(), namespace, selector.String(), timeout, sortBy)
	return pod, err
}
