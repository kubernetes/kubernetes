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

package apiclient

import (
	"context"
	"encoding/json"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// objectMutator is a function that mutates the given runtime object and optionally returns an error
type objectMutator[T runtime.Object] func(T) error

// apiCallRetryInterval holds a local copy of apiCallRetryInterval for testing purposes
var apiCallRetryInterval = constants.KubernetesAPICallRetryInterval

type kubernetesInterface[T kubernetesObject] interface {
	Create(ctx context.Context, obj T, opts metav1.CreateOptions) (T, error)
	Get(ctx context.Context, name string, opts metav1.GetOptions) (T, error)
	Update(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error)
}

type kubernetesObject interface {
	runtime.Object
	metav1.Object
}

// CreateOrUpdate creates a runtime object if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
func CreateOrUpdate[T kubernetesObject](client kubernetesInterface[T], obj T) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			// This uses a background context for API calls to avoid confusing callers that don't
			// expect context-related errors.
			ctx := context.Background()
			if _, err := client.Create(ctx, obj, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrapf(err, "unable to create %T", obj)
					return false, nil
				}
				if _, err := client.Update(ctx, obj, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrapf(err, "unable to update %T", obj)
					return false, nil
				}
			}
			return true, nil
		})
	if err == nil {
		return nil
	}
	return lastError
}

// CreateOrMutate tries to create the provided object. If the resource exists already, the latest version will be fetched from
// the cluster and mutator callback will be called on it, then an Update of the mutated object will be performed. This function is resilient
// to conflicts, and a retry will be issued if the object was modified on the server between the refresh and the update (while the mutation was
// taking place).
func CreateOrMutate[T kubernetesObject](client kubernetesInterface[T], obj T, mutator objectMutator[T]) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			// This uses a background context for API calls to avoid confusing callers that don't
			// expect context-related errors.
			ctx := context.Background()
			if _, err := client.Create(ctx, obj, metav1.CreateOptions{}); err != nil {
				lastError = err
				if apierrors.IsAlreadyExists(err) {
					lastError = mutate(ctx, client, metav1.ObjectMeta{Namespace: obj.GetNamespace(), Name: obj.GetName()}, mutator)
					return lastError == nil, nil
				}
				return false, nil
			}
			return true, nil
		})
	if err == nil {
		return nil
	}
	return lastError
}

// mutate takes an Object Meta (namespace and name), retrieves the resource from the server and tries to mutate it
// by calling to the mutator callback, then an Update of the mutated object will be performed. This function is resilient
// to conflicts, and a retry will be issued if the object was modified on the server between the refresh and the update (while the mutation was
// taking place).
func mutate[T kubernetesObject](ctx context.Context, client kubernetesInterface[T], meta metav1.ObjectMeta, mutator objectMutator[T]) error {
	obj, err := client.Get(ctx, meta.Name, metav1.GetOptions{})
	if err != nil {
		return errors.Wrapf(err, "unable to get %T", obj)
	}
	if err = mutator(obj); err != nil {
		return errors.Wrapf(err, "unable to mutate %T", obj)
	}
	_, err = client.Update(ctx, obj, metav1.UpdateOptions{})
	return err
}

// CreateOrRetain creates a runtime object if the target resource doesn't exist.
// If the resource exists already, this function will retain the resource instead.
func CreateOrRetain[T kubernetesObject](client kubernetesInterface[T], obj T, name string) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			// This uses a background context for API calls to avoid confusing callers that don't
			// expect context-related errors.
			ctx := context.Background()
			if _, err := client.Get(ctx, name, metav1.GetOptions{}); err != nil {
				if !apierrors.IsNotFound(err) {
					lastError = errors.Wrapf(err, "unable to get %T", obj)
					return false, nil
				}
				if _, err := client.Create(ctx, obj, metav1.CreateOptions{}); err != nil {
					lastError = errors.Wrapf(err, "unable to create %T", obj)
					return false, nil
				}
			}
			return true, nil
		})
	if err == nil {
		return nil
	}
	return lastError
}

// PatchNodeOnce executes patchFn on the node object found by the node name.
func PatchNodeOnce(client clientset.Interface, nodeName string, patchFn func(*v1.Node), lastError *error) func(context.Context) (bool, error) {
	return func(_ context.Context) (bool, error) {
		// First get the node object
		ctx := context.Background()
		n, err := client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			*lastError = err
			return false, nil // retry on any error
		}

		// The node may appear to have no labels at first,
		// so we wait for it to get hostname label.
		if _, found := n.ObjectMeta.Labels[v1.LabelHostname]; !found {
			return false, nil
		}

		oldData, err := json.Marshal(n)
		if err != nil {
			*lastError = errors.Wrapf(err, "failed to marshal unmodified node %q into JSON", n.Name)
			return false, *lastError
		}

		// Execute the mutating function
		patchFn(n)

		newData, err := json.Marshal(n)
		if err != nil {
			*lastError = errors.Wrapf(err, "failed to marshal modified node %q into JSON", n.Name)
			return false, *lastError
		}

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		if err != nil {
			*lastError = errors.Wrap(err, "failed to create two way merge patch")
			return false, *lastError
		}

		if _, err := client.CoreV1().Nodes().Patch(ctx, n.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}); err != nil {
			*lastError = errors.Wrapf(err, "error patching Node %q", n.Name)
			return false, nil
		}

		return true, nil
	}
}

// PatchNode tries to patch a node using patchFn for the actual mutating logic.
// Retries are provided by the wait package.
func PatchNode(client clientset.Interface, nodeName string, patchFn func(*v1.Node)) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, PatchNodeOnce(client, nodeName, patchFn, &lastError))
	if err == nil {
		return nil
	}
	return lastError
}

// GetConfigMapWithShortRetry tries to retrieve a ConfigMap using the given client, retrying for a short
// time if it gets an unexpected error. The main usage of this function is in areas of the code that
// fallback to a default ConfigMap value in case the one from the API cannot be quickly obtained.
func GetConfigMapWithShortRetry(client clientset.Interface, namespace, name string) (*v1.ConfigMap, error) {
	var cm *v1.ConfigMap
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		time.Millisecond*50, time.Millisecond*350,
		true, func(_ context.Context) (bool, error) {
			var err error
			// Intentionally pass a new context to this API call. This will let the API call run
			// independently of the parent context timeout, which is quite short and can cause the API
			// call to return abruptly.
			cm, err = client.CoreV1().ConfigMaps(namespace).Get(context.Background(), name, metav1.GetOptions{})
			if err == nil {
				return true, nil
			}
			lastError = err
			return false, nil
		})
	if err == nil {
		return cm, nil
	}
	return nil, lastError
}
