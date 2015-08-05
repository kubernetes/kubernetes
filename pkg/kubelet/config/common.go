/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Common logic used by both http and file channels.
package config

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"

	"github.com/golang/glog"
)

// Generate a pod name that is unique among nodes by appending the nodeName.
func generatePodName(name, nodeName string) string {
	return fmt.Sprintf("%s-%s", name, nodeName)
}

func applyDefaults(pod *api.Pod, source string, isFile bool, nodeName string) error {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		if isFile {
			fmt.Fprintf(hasher, "host:%s", nodeName)
			fmt.Fprintf(hasher, "file:%s", source)
		} else {
			fmt.Fprintf(hasher, "url:%s", source)
		}
		util.DeepHashObject(hasher, pod)
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		glog.V(5).Infof("Generated UID %q pod %q from %s", pod.UID, pod.Name, source)
	}

	pod.Name = generatePodName(pod.Name, nodeName)
	glog.V(5).Infof("Generated Name %q for UID %q from URL %s", pod.Name, pod.UID, source)

	if pod.Namespace == "" {
		pod.Namespace = kubelet.NamespaceDefault
	}
	glog.V(5).Infof("Using namespace %q for pod %q from %s", pod.Namespace, pod.Name, source)

	// Set the Host field to indicate this pod is scheduled on the current node.
	pod.Spec.NodeName = nodeName

	pod.ObjectMeta.SelfLink = getSelfLink(pod.Name, pod.Namespace)
	return nil
}

func getSelfLink(name, namespace string) string {
	var selfLink string
	if len(namespace) == 0 {
		namespace = api.NamespaceDefault
	}
	selfLink = fmt.Sprintf("/api/"+latest.Version+"/pods/namespaces/%s/%s", name, namespace)
	return selfLink
}

type defaultFunc func(pod *api.Pod) error

func tryDecodeSinglePod(data []byte, defaultFn defaultFunc) (parsed bool, pod *api.Pod, err error) {
	// JSON is valid YAML, so this should work for everything.
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return false, nil, err
	}
	obj, err := api.Scheme.Decode(json)
	if err != nil {
		return false, pod, err
	}
	// Check whether the object could be converted to single pod.
	if _, ok := obj.(*api.Pod); !ok {
		err = fmt.Errorf("invalid pod: %+v", obj)
		return false, pod, err
	}
	newPod := obj.(*api.Pod)
	// Apply default values and validate the pod.
	if err = defaultFn(newPod); err != nil {
		return true, pod, err
	}
	if errs := validation.ValidatePod(newPod); len(errs) > 0 {
		err = fmt.Errorf("invalid pod: %v", errs)
		return true, pod, err
	}
	return true, newPod, nil
}

func tryDecodePodList(data []byte, defaultFn defaultFunc) (parsed bool, pods api.PodList, err error) {
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return false, api.PodList{}, err
	}
	obj, err := api.Scheme.Decode(json)
	if err != nil {
		return false, pods, err
	}
	// Check whether the object could be converted to list of pods.
	if _, ok := obj.(*api.PodList); !ok {
		err = fmt.Errorf("invalid pods list: %+v", obj)
		return false, pods, err
	}
	newPods := obj.(*api.PodList)
	// Apply default values and validate pods.
	for i := range newPods.Items {
		newPod := &newPods.Items[i]
		if err = defaultFn(newPod); err != nil {
			return true, pods, err
		}
		if errs := validation.ValidatePod(newPod); len(errs) > 0 {
			err = fmt.Errorf("invalid pod: %v", errs)
			return true, pods, err
		}
	}
	return true, *newPods, err
}
