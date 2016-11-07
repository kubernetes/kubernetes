/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"hash/adler32"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/retry"
	errorsutil "k8s.io/kubernetes/pkg/util/errors"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/pkg/util/wait"
)

func GetPodTemplateSpecHash(template v1.PodTemplateSpec) uint32 {
	podTemplateSpecHasher := adler32.New()
	hashutil.DeepHashObject(podTemplateSpecHasher, template)
	return podTemplateSpecHasher.Sum32()
}

// TODO: remove the duplicate
func GetInternalPodTemplateSpecHash(template api.PodTemplateSpec) uint32 {
	podTemplateSpecHasher := adler32.New()
	hashutil.DeepHashObject(podTemplateSpecHasher, template)
	return podTemplateSpecHasher.Sum32()
}

// TODO: use client library instead when it starts to support update retries
//       see https://github.com/kubernetes/kubernetes/issues/21479
type updatePodFunc func(pod *v1.Pod) error

// UpdatePodWithRetries updates a pod with given applyUpdate function. Note that pod not found error is ignored.
// The returned bool value can be used to tell if the pod is actually updated.
// TODO: Switch this to PATCH? We definitely want adoption to succeed but this is a generic helper.
func UpdatePodWithRetries(podClient v1core.PodInterface, podLister *cache.StoreToPodLister, namespace, name string, applyUpdate updatePodFunc) (*v1.Pod, error) {
	var err error
	var pod *v1.Pod

	pollErr := wait.ExponentialBackoff(retry.DefaultBackoff, func() (bool, error) {
		var getErr error
		pod, getErr := podLister.Pods(namespace).Get(name)
		if getErr != nil {
			return false, getErr
		}
		obj, deepCopyErr := api.Scheme.DeepCopy(pod)
		if deepCopyErr != nil {
			return false, deepCopyErr
		}
		pod = obj.(*v1.Pod)
		// Apply the update, then attempt to push it to the apiserver.
		if applyErr := applyUpdate(pod); applyErr != nil {
			return false, applyErr
		}
		pod, err = podClient.Update(pod)
		if err == nil {
			// Update successful.
			return true, nil
		}
		if !errors.IsConflict(err) {
			// Do not retry if the error is not an update conflict.
			return false, err
		}
		// Update could have failed due to conflict error. Try again.
		return false, nil
	})

	// Handle returned error from wait poll
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("timed out trying to update pod %q: %v", name, err)
	}
	// Ignore the precondition violated error, this pod is already updated
	// with the desired label.
	if pollErr == errorsutil.ErrPreconditionViolated {
		glog.V(4).Infof("Pod %s/%s precondition doesn't hold, skip updating it.", namespace, name)
		pollErr = nil
	}

	// If pollErr is non-nil the returned pod cannot be trusted.
	// Otherwise the returned pod contains the applied update.
	return pod, pollErr
}
