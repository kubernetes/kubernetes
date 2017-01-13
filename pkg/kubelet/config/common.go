/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/validation"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/hash"

	"github.com/golang/glog"
)

// Generate a pod name that is unique among nodes by appending the nodeName.
func generatePodName(name string, nodeName types.NodeName) string {
	return fmt.Sprintf("%s-%s", name, nodeName)
}

func applyDefaults(pod *api.Pod, source string, isFile bool, nodeName types.NodeName) error {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		if isFile {
			fmt.Fprintf(hasher, "host:%s", nodeName)
			fmt.Fprintf(hasher, "file:%s", source)
		} else {
			fmt.Fprintf(hasher, "url:%s", source)
		}
		hash.DeepHashObject(hasher, pod)
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		glog.V(5).Infof("Generated UID %q pod %q from %s", pod.UID, pod.Name, source)
	}

	pod.Name = generatePodName(pod.Name, nodeName)
	glog.V(5).Infof("Generated Name %q for UID %q from URL %s", pod.Name, pod.UID, source)

	if pod.Namespace == "" {
		pod.Namespace = kubetypes.NamespaceDefault
	}
	glog.V(5).Infof("Using namespace %q for pod %q from %s", pod.Namespace, pod.Name, source)

	// Set the Host field to indicate this pod is scheduled on the current node.
	pod.Spec.NodeName = string(nodeName)

	pod.ObjectMeta.SelfLink = getSelfLink(pod.Name, pod.Namespace)

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	// The generated UID is the hash of the file.
	pod.Annotations[kubetypes.ConfigHashAnnotationKey] = string(pod.UID)

	// Set the default status to pending.
	pod.Status.Phase = api.PodPending
	return nil
}

func getSelfLink(name, namespace string) string {
	var selfLink string
	if len(namespace) == 0 {
		namespace = api.NamespaceDefault
	}
	selfLink = fmt.Sprintf("/api/"+api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version+"/pods/namespaces/%s/%s", name, namespace)
	return selfLink
}

type defaultFunc func(pod *api.Pod) error

func tryDecodeSinglePod(data []byte, defaultFn defaultFunc) (parsed bool, pod *v1.Pod, err error) {
	// JSON is valid YAML, so this should work for everything.
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return false, nil, err
	}
	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(), json)
	if err != nil {
		return false, pod, err
	}
	// Check whether the object could be converted to single pod.
	if _, ok := obj.(*api.Pod); !ok {
		err = fmt.Errorf("invalid pod: %#v", obj)
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
	v1Pod := &v1.Pod{}
	if err := v1.Convert_api_Pod_To_v1_Pod(newPod, v1Pod, nil); err != nil {
		return true, nil, err
	}
	return true, v1Pod, nil
}

func tryDecodePodList(data []byte, defaultFn defaultFunc) (parsed bool, pods v1.PodList, err error) {
	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(), data)
	if err != nil {
		return false, pods, err
	}
	// Check whether the object could be converted to list of pods.
	if _, ok := obj.(*api.PodList); !ok {
		err = fmt.Errorf("invalid pods list: %#v", obj)
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
	v1Pods := &v1.PodList{}
	if err := v1.Convert_api_PodList_To_v1_PodList(newPods, v1Pods, nil); err != nil {
		return true, pods, err
	}
	return true, *v1Pods, err
}
