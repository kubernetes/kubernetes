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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilyaml "github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

// Generate a pod name that is unique among nodes by appending the hostname.
func generatePodName(name, hostname string) string {
	return fmt.Sprintf("%s-%s", name, hostname)
}

func applyDefaults(pod *api.Pod, source string, isFile bool, hostname string) error {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		if isFile {
			fmt.Fprintf(hasher, "host:%s", hostname)
			fmt.Fprintf(hasher, "file:%s", source)
		} else {
			fmt.Fprintf(hasher, "url:%s", source)
		}
		util.DeepHashObject(hasher, pod)
		pod.UID = types.UID(hex.EncodeToString(hasher.Sum(nil)[0:]))
		glog.V(5).Infof("Generated UID %q pod %q from %s", pod.UID, pod.Name, source)
	}

	// This is required for backward compatibility, and should be removed once we
	// completely deprecate ContainerManifest.
	if len(pod.Name) == 0 {
		pod.Name = string(pod.UID)
	}
	pod.Name = generatePodName(pod.Name, hostname)
	glog.V(5).Infof("Generated Name %q for UID %q from URL %s", pod.Name, pod.UID, source)

	if pod.Namespace == "" {
		pod.Namespace = kubelet.NamespaceDefault
	}
	glog.V(5).Infof("Using namespace %q for pod %q from %s", pod.Namespace, pod.Name, source)

	// Set the Host field to indicate this pod is scheduled on the current node.
	pod.Spec.Host = hostname

	pod.ObjectMeta.SelfLink = getSelfLink(pod.Name, pod.Namespace)
	return nil
}

func getSelfLink(name, namespace string) string {
	var selfLink string
	if api.PreV1Beta3(latest.Version) {
		selfLink = fmt.Sprintf("/api/"+latest.Version+"/pods/%s?namespace=%s", name, namespace)
	} else {
		if len(namespace) == 0 {
			namespace = api.NamespaceDefault
		}
		selfLink = fmt.Sprintf("/api/"+latest.Version+"/pods/namespaces/%s/%s", name, namespace)
	}
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

func tryDecodeSingleManifest(data []byte, defaultFn defaultFunc) (parsed bool, manifest v1beta1.ContainerManifest, pod *api.Pod, err error) {
	// TODO: should be api.Scheme.Decode
	// This is awful.  DecodeInto() expects to find an APIObject, which
	// Manifest is not.  We keep reading manifest for now for compat, but
	// we will eventually change it to read Pod (at which point this all
	// becomes nicer).  Until then, we assert that the ContainerManifest
	// structure on disk is always v1beta1.  Read that, convert it to a
	// "current" ContainerManifest (should be ~identical), then convert
	// that to a Pod (which is a well-understood conversion).  This
	// avoids writing a v1beta1.ContainerManifest -> api.Pod
	// conversion which would be identical to the api.ContainerManifest ->
	// api.Pod conversion.
	pod = new(api.Pod)
	if err = yaml.Unmarshal(data, &manifest); err != nil {
		return false, manifest, pod, err
	}
	newManifest := api.ContainerManifest{}
	if err = api.Scheme.Convert(&manifest, &newManifest); err != nil {
		return false, manifest, pod, err
	}
	if errs := validation.ValidateManifest(&newManifest); len(errs) > 0 {
		err = fmt.Errorf("invalid manifest: %v", errs)
		return false, manifest, pod, err
	}
	if err = api.Scheme.Convert(&newManifest, pod); err != nil {
		return true, manifest, pod, err
	}
	if err := defaultFn(pod); err != nil {
		return true, manifest, pod, err
	}
	// Success.
	return true, manifest, pod, nil
}

func tryDecodeManifestList(data []byte, defaultFn defaultFunc) (parsed bool, manifests []v1beta1.ContainerManifest, pods api.PodList, err error) {
	// TODO: should be api.Scheme.Decode
	// See the comment in tryDecodeSingle().
	if err = yaml.Unmarshal(data, &manifests); err != nil {
		return false, manifests, pods, err
	}
	newManifests := []api.ContainerManifest{}
	if err = api.Scheme.Convert(&manifests, &newManifests); err != nil {
		return false, manifests, pods, err
	}
	for i := range newManifests {
		manifest := &newManifests[i]
		if errs := validation.ValidateManifest(manifest); len(errs) > 0 {
			err = fmt.Errorf("invalid manifest: %v", errs)
			return false, manifests, pods, err
		}
	}
	list := api.ContainerManifestList{Items: newManifests}
	if err = api.Scheme.Convert(&list, &pods); err != nil {
		return true, manifests, pods, err
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if err := defaultFn(pod); err != nil {
			return true, manifests, pods, err
		}
	}
	// Success.
	return true, manifests, pods, nil
}
