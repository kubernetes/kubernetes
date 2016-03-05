/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package pod

import (
	"fmt"
	"hash/adler32"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	errorsutil "k8s.io/kubernetes/pkg/util/errors"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
	"k8s.io/kubernetes/pkg/util/wait"
)

func GetPodTemplateSpecHash(template api.PodTemplateSpec) uint32 {
	podTemplateSpecHasher := adler32.New()
	hashutil.DeepHashObject(podTemplateSpecHasher, template)
	return podTemplateSpecHasher.Sum32()
}

// TODO: use client library instead when it starts to support update retries
//       see https://github.com/kubernetes/kubernetes/issues/21479
type updatePodFunc func(pod *api.Pod) error

// UpdatePodWithRetries updates a pod with given applyUpdate function. Note that pod not found error is ignored.
// The returned bool value can be used to tell if the pod is actually updated.
func UpdatePodWithRetries(podClient unversionedcore.PodInterface, pod *api.Pod, applyUpdate updatePodFunc) (*api.Pod, bool, error) {
	var err error
	var podUpdated bool
	oldPod := pod
	if err = wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		pod, err = podClient.Get(oldPod.Name)
		if err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		if err = applyUpdate(pod); err != nil {
			return false, err
		}
		if pod, err = podClient.Update(pod); err == nil {
			// Update successful.
			return true, nil
		}
		// TODO: don't retry on perm-failed errors and handle them gracefully
		// Update could have failed due to conflict error. Try again.
		return false, nil
	}); err == nil {
		// When there's no error, we've updated this pod.
		podUpdated = true
	}

	// Handle returned error from wait poll
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("timed out trying to update pod: %+v", oldPod)
	}
	// Ignore the pod not found error, but the pod isn't updated.
	if errors.IsNotFound(err) {
		glog.V(4).Infof("%s %s/%s is not found, skip updating it.", oldPod.Kind, oldPod.Namespace, oldPod.Name)
		err = nil
	}
	// Ignore the precondition violated error, but the pod isn't updated.
	if err == errorsutil.ErrPreconditionViolated {
		glog.V(4).Infof("%s %s/%s precondition doesn't hold, skip updating it.", oldPod.Kind, oldPod.Namespace, oldPod.Name)
		err = nil
	}

	// If the error is non-nil the returned pod cannot be trusted; if podUpdated is false, the pod isn't updated;
	// if the error is nil and podUpdated is true, the returned pod contains the applied update.
	return pod, podUpdated, err
}
