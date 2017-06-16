/*
Copyright 2017 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
)

type deleteFunc func(options *metav1.DeleteOptions) (runtime.Object, error)

// Sends a request to DELETE the object and waits till the object is deleted.
// deleteFn should return the deleted object or a metav1.Status object.
func WaitForDeletion(deleteFn deleteFunc, options *metav1.DeleteOptions, timeout time.Duration) error {
	obj, err := deleteFn(options)
	if err != nil {
		return err
	}
	// Keep deleting the resource with uid precondition until we get a
	// resource not found error.
	// We use DELETE here instead of using GET to ensure that DELETE
	// works without requiring GET permissions.
	uid, err := getUID(obj)
	if err != nil {
		return fmt.Errorf("unexpected error in extracting uid: %s", err)
	}
	if uid == "" {
		// For backwards compatibility, we just return when we are unable to get the UID.
		// TODO: Remove this in 1.8.
		return nil
	}
	options.Preconditions = metav1.NewUIDPreconditions(uid)
	err = wait.PollImmediate(time.Second, timeout, func() (bool, error) {
		_, err := deleteFn(options)
		if errors.IsNotFound(err) || errors.IsConflict(err) {
			// Resource successfully deleted.
			return true, nil
		}
		return false, err
	})
	return err
}

func getUID(obj runtime.Object) (string, error) {
	// Check if the object is of type Status.
	status, isStatus := obj.(*metav1.Status)
	if isStatus {
		if status.Details == nil || status.Details.UID == "" {
			// This can happen for an older apiserver (before 1.7)
			// which does not set Details.
			// For now, we do not generate an error in this case to ensure
			// kubectl can work with 1.6 apiserver.
			// TODO: In 1.8, update this to generate an error.
			return "", nil
		}
		return string(status.Details.UID), nil
	}
	// The resource is itself returned.
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	return string(accessor.GetUID()), nil
}

// TODO: Update the generated clientset to return the deleted object and then update the callers to use that.
func DeleteWithRestClient(client RESTClient, name, namespace, resource string, isNamespaced bool, options *metav1.DeleteOptions) (runtime.Object, error) {
	return client.Delete().
		NamespaceIfScoped(namespace, isNamespaced).
		Resource(resource).
		Name(name).
		Body(options).
		Do().
		Get()
}
