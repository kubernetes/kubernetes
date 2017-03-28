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
	"hash/adler32"
	"hash/fnv"

	"github.com/golang/glog"

	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/client/retry"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
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

func GetPodTemplateSpecHashFnv(template v1.PodTemplateSpec) uint32 {
	podTemplateSpecHasher := fnv.New32a()
	hashutil.DeepHashObject(podTemplateSpecHasher, template)
	return podTemplateSpecHasher.Sum32()
}

// TODO: use client library instead when it starts to support update retries
//       see https://github.com/kubernetes/kubernetes/issues/21479
type updatePodFunc func(pod *v1.Pod) error

// UpdatePodWithRetries updates a pod with given applyUpdate function. Note that pod not found error is ignored.
// The returned bool value can be used to tell if the pod is actually updated.
func UpdatePodWithRetries(podClient v1core.PodInterface, podLister corelisters.PodLister, namespace, name string, applyUpdate updatePodFunc) (*v1.Pod, error) {
	var pod *v1.Pod

	retryErr := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		var err error
		pod, err = podLister.Pods(namespace).Get(name)
		if err != nil {
			return err
		}
		obj, deepCopyErr := api.Scheme.DeepCopy(pod)
		if deepCopyErr != nil {
			return deepCopyErr
		}
		pod = obj.(*v1.Pod)
		// Apply the update, then attempt to push it to the apiserver.
		if applyErr := applyUpdate(pod); applyErr != nil {
			return applyErr
		}
		pod, err = podClient.Update(pod)
		return err
	})

	// Ignore the precondition violated error, this pod is already updated
	// with the desired label.
	if retryErr == errorsutil.ErrPreconditionViolated {
		glog.V(4).Infof("Pod %s/%s precondition doesn't hold, skip updating it.", namespace, name)
		retryErr = nil
	}

	return pod, retryErr
}
