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

package config

import (
	"encoding/hex"
	"errors"
	"fmt"
	"hash/fnv"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/features"

	// TODO: remove this import if
	// api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String() is changed
	// to "v1"?
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	// Ensure that core apis are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/hash"

	"k8s.io/klog/v2"
)

const (
	maxConfigLength = 10 * 1 << 20 // 10MB
)

// Generate a pod name that is unique among nodes by appending the nodeName.
func generatePodName(name string, nodeName types.NodeName) string {
	return fmt.Sprintf("%s-%s", name, strings.ToLower(string(nodeName)))
}

func applyDefaults(logger klog.Logger, pod *api.Pod, source string, isFile bool, nodeName types.NodeName) error {
	if len(pod.UID) == 0 {
		hasher := fnv.New128a()
		hash.DeepHashObject(hasher, pod)
		// DeepHashObject resets the hash, so we should write the pod source
		// information AFTER it.
		if isFile {
			fmt.Fprintf(hasher, "host:%s", nodeName)
			fmt.Fprintf(hasher, "file:%s", source)
		} else {
			fmt.Fprintf(hasher, "url:%s", source)
		}
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		logger.V(5).Info("Generated UID", "pod", klog.KObj(pod), "podUID", pod.UID, "source", source)
	}

	pod.Name = generatePodName(pod.Name, nodeName)
	logger.V(5).Info("Generated pod name", "pod", klog.KObj(pod), "podUID", pod.UID, "source", source)

	if pod.Namespace == "" {
		pod.Namespace = metav1.NamespaceDefault
	}
	logger.V(5).Info("Set namespace for pod", "pod", klog.KObj(pod), "source", source)

	// Set the Host field to indicate this pod is scheduled on the current node.
	pod.Spec.NodeName = string(nodeName)

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	// The generated UID is the hash of the file.
	pod.Annotations[kubetypes.ConfigHashAnnotationKey] = string(pod.UID)

	if isFile {
		// Applying the default Taint tolerations to static pods,
		// so they are not evicted when there are node problems.
		helper.AddOrUpdateTolerationInPod(pod, &api.Toleration{
			Operator: "Exists",
			Effect:   api.TaintEffectNoExecute,
		})
	}

	// Set the default status to pending.
	pod.Status.Phase = api.PodPending
	return nil
}

type defaultFunc func(logger klog.Logger, pod *api.Pod) error

// A static pod tried to use a ClusterTrustBundle projected volume source.
var ErrStaticPodTriedToUseClusterTrustBundle = errors.New("static pods may not use ClusterTrustBundle projected volume sources")

// A static pod tried to use a resource claim.
var ErrStaticPodTriedToUseResourceClaims = errors.New("static pods may not use ResourceClaims")

// tryDecodeSinglePod takes data and tries to extract valid Pod config information from it.
func tryDecodeSinglePod(logger klog.Logger, data []byte, defaultFn defaultFunc) (parsed bool, pod *v1.Pod, err error) {
	// JSON is valid YAML, so this should work for everything.
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return false, nil, err
	}
	obj, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), json)
	if err != nil {
		return false, pod, err
	}

	newPod, ok := obj.(*api.Pod)
	// Check whether the object could be converted to single pod.
	if !ok {
		return false, pod, fmt.Errorf("invalid pod: %#v", obj)
	}

	if newPod.Name == "" {
		return true, pod, fmt.Errorf("invalid pod: name is needed for the pod")
	}

	// Apply default values and validate the pod.
	if err = defaultFn(logger, newPod); err != nil {
		return true, pod, err
	}
	opts := podutil.GetValidationOptionsFromPodSpecAndMeta(&newPod.Spec, nil, &newPod.ObjectMeta, nil)
	if errs := validation.ValidatePodCreate(newPod, opts); len(errs) > 0 {
		return true, pod, fmt.Errorf("invalid pod: %v", errs)
	}
	v1Pod := &v1.Pod{}
	if err := k8s_api_v1.Convert_core_Pod_To_v1_Pod(newPod, v1Pod, nil); err != nil {
		logger.Error(err, "Pod failed to convert to v1", "pod", klog.KObj(newPod))
		return true, nil, err
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.PreventStaticPodAPIReferences) {
		// Check if pod has references to API objects
		_, resource, err := podutil.HasAPIObjectReference(newPod)
		if err != nil {
			return true, nil, err
		}
		if resource != "" {
			return true, nil, fmt.Errorf("static pods may not reference %s", resource)
		}
	} else {
		// TODO: Remove this else block once the PreventStaticPodAPIReferences gate is GA
		for _, v := range v1Pod.Spec.Volumes {
			if v.Projected == nil {
				continue
			}

			for _, s := range v.Projected.Sources {
				if s.ClusterTrustBundle != nil {
					return true, nil, ErrStaticPodTriedToUseClusterTrustBundle
				}
			}
		}
		if len(v1Pod.Spec.ResourceClaims) > 0 {
			return true, nil, ErrStaticPodTriedToUseResourceClaims
		}
	}

	return true, v1Pod, nil
}

func tryDecodePodList(logger klog.Logger, data []byte, defaultFn defaultFunc) (parsed bool, pods v1.PodList, err error) {
	obj, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		return false, pods, err
	}

	newPods, ok := obj.(*api.PodList)
	// Check whether the object could be converted to list of pods.
	if !ok {
		err = fmt.Errorf("invalid pods list: %#v", obj)
		return false, pods, err
	}

	// Apply default values and validate pods.
	for i := range newPods.Items {
		newPod := &newPods.Items[i]
		if newPod.Name == "" {
			return true, pods, fmt.Errorf("invalid pod: name is needed for the pod")
		}
		if err = defaultFn(logger, newPod); err != nil {
			return true, pods, err
		}
		opts := podutil.GetValidationOptionsFromPodSpecAndMeta(&newPod.Spec, nil, &newPod.ObjectMeta, nil)
		if errs := validation.ValidatePodCreate(newPod, opts); len(errs) > 0 {
			err = fmt.Errorf("invalid pod: %v", errs)
			return true, pods, err
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.PreventStaticPodAPIReferences) {
			// Check if pod has references to API objects
			_, resource, err := podutil.HasAPIObjectReference(newPod)
			if err != nil {
				return true, pods, err
			}
			if resource != "" {
				return true, pods, fmt.Errorf("static pods may not reference %s", resource)
			}
		}
	}
	v1Pods := &v1.PodList{}
	if err := k8s_api_v1.Convert_core_PodList_To_v1_PodList(newPods, v1Pods, nil); err != nil {
		return true, pods, err
	}
	return true, *v1Pods, err
}
