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

// Common logic used by both http and file channels.
package config

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"os"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

func applyDefaults(pod *api.Pod, source string, isFile bool) error {
	if len(pod.UID) == 0 {
		hasher := md5.New()
		if isFile {
			hostname, err := os.Hostname() // TODO: kubelet name would be better
			if err != nil {
				return err
			}
			hostname = strings.ToLower(hostname)
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
	var err error
	if len(pod.Name) == 0 {
		pod.Name = string(pod.UID)
	}
	if pod.Name, err = GeneratePodName(pod.Name); err != nil {
		return err
	}
	glog.V(5).Infof("Generated Name %q for UID %q from URL %s", pod.Name, pod.UID, source)

	if pod.Namespace == "" {
		pod.Namespace = kubelet.NamespaceDefault
	}
	glog.V(5).Infof("Using namespace %q for pod %q from %s", pod.Namespace, pod.Name, source)

	// Currently just simply follow the same format in resthandler.go
	pod.ObjectMeta.SelfLink =
		fmt.Sprintf("/api/v1beta2/pods/%s?namespace=%s", pod.Name, pod.Namespace)
	return nil
}

func tryDecodeSinglePod(data []byte, source string, isFile bool) (parsed bool, pod api.Pod, err error) {
	obj, err := api.Scheme.Decode(data)
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
	if err = applyDefaults(newPod, source, isFile); err != nil {
		return true, pod, err
	}
	if errs := validation.ValidatePod(newPod); len(errs) > 0 {
		err = fmt.Errorf("invalid pod: %v", errs)
		return true, pod, err
	}
	return true, *newPod, nil
}

func tryDecodePodList(data []byte, source string, isFile bool) (parsed bool, pods api.PodList, err error) {
	obj, err := api.Scheme.Decode(data)
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
		if err = applyDefaults(newPod, source, isFile); err != nil {
			return true, pods, err
		}
		if errs := validation.ValidatePod(newPod); len(errs) > 0 {
			err = fmt.Errorf("invalid pod: %v", errs)
			return true, pods, err
		}
	}
	return true, *newPods, err
}
