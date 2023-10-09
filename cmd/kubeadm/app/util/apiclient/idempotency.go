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

	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
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
func CreateOrUpdate[T kubernetesObject](ctx context.Context, client kubernetesInterface[T], obj T) error {
	var lastError error
	err := wait.PollUntilContextTimeout(ctx,
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

// CreateOrUpdateConfigMap creates a ConfigMap if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateConfigMap(client clientset.Interface, cm *v1.ConfigMap) error {
	return CreateOrUpdate(context.Background(), client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace), cm)
}

// CreateOrMutate tries to create the provided object. If the resource exists already, the latest version will be fetched from
// the cluster and mutator callback will be called on it, then an Update of the mutated object will be performed. This function is resilient
// to conflicts, and a retry will be issued if the object was modified on the server between the refresh and the update (while the mutation was
// taking place).
func CreateOrMutate[T kubernetesObject](ctx context.Context, client kubernetesInterface[T], obj T, mutator objectMutator[T]) error {
	var lastError error
	err := wait.PollUntilContextTimeout(ctx,
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

// CreateOrMutateConfigMap tries to create the ConfigMap provided as cm. If the resource exists already, the latest version will be fetched from
// the cluster and mutator callback will be called on it, then an Update of the mutated ConfigMap will be performed. This function is resilient
// to conflicts, and a retry will be issued if the ConfigMap was modified on the server between the refresh and the update (while the mutation was
// taking place).
//
// Deprecated: use CreateOrMutate() instead.
func CreateOrMutateConfigMap(client clientset.Interface, cm *v1.ConfigMap, mutator objectMutator[*v1.ConfigMap]) error {
	return CreateOrMutate(context.Background(), client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace), cm, mutator)
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
func CreateOrRetain[T kubernetesObject](ctx context.Context, client kubernetesInterface[T], obj T) error {
	var lastError error
	err := wait.PollUntilContextTimeout(ctx,
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			// This uses a background context for API calls to avoid confusing callers that don't
			// expect context-related errors.
			ctx := context.Background()
			if _, err := client.Get(ctx, obj.GetName(), metav1.GetOptions{}); err != nil {
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

// CreateOrRetainConfigMap creates a ConfigMap if the target resource doesn't exist.
// If the resource exists already, this function will retain the resource instead.
//
// Deprecated: use CreateOrRetain() instead.
func CreateOrRetainConfigMap(client clientset.Interface, cm *v1.ConfigMap, configMapName string) error {
	return CreateOrRetain(context.Background(), client.CoreV1().ConfigMaps(cm.Namespace), cm)
}

// CreateOrUpdateSecret creates a Secret if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateSecret(client clientset.Interface, secret *v1.Secret) error {
	return CreateOrUpdate(context.Background(), client.CoreV1().Secrets(secret.Namespace), secret)
}

// CreateOrUpdateServiceAccount creates a ServiceAccount if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateServiceAccount(client clientset.Interface, sa *v1.ServiceAccount) error {
	return CreateOrUpdate(context.Background(), client.CoreV1().ServiceAccounts(sa.Namespace), sa)
}

// CreateOrUpdateDeployment creates a Deployment if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateDeployment(client clientset.Interface, deploy *apps.Deployment) error {
	return CreateOrUpdate(context.Background(), client.AppsV1().Deployments(deploy.Namespace), deploy)
}

// CreateOrRetainDeployment creates a Deployment if the target resource doesn't exist.
// If the resource exists already, this function will retain the resource instead.
//
// Deprecated: use CreateOrRetain() instead.
func CreateOrRetainDeployment(client clientset.Interface, deploy *apps.Deployment, deployName string) error {
	return CreateOrRetain(context.Background(), client.AppsV1().Deployments(deploy.Namespace), deploy)
}

// CreateOrUpdateDaemonSet creates a DaemonSet if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateDaemonSet(client clientset.Interface, ds *apps.DaemonSet) error {
	return CreateOrUpdate(context.Background(), client.AppsV1().DaemonSets(ds.Namespace), ds)
}

// CreateOrUpdateRole creates a Role if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateRole(client clientset.Interface, role *rbac.Role) error {
	return CreateOrUpdate(context.Background(), client.RbacV1().Roles(role.Namespace), role)
}

// CreateOrUpdateRoleBinding creates a RoleBinding if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateRoleBinding(client clientset.Interface, roleBinding *rbac.RoleBinding) error {
	return CreateOrUpdate(context.Background(), client.RbacV1().RoleBindings(roleBinding.Namespace), roleBinding)
}

// CreateOrUpdateClusterRole creates a ClusterRole if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateClusterRole(client clientset.Interface, clusterRole *rbac.ClusterRole) error {
	return CreateOrUpdate(context.Background(), client.RbacV1().ClusterRoles(), clusterRole)
}

// CreateOrUpdateClusterRoleBinding creates a ClusterRoleBinding if the target resource doesn't exist.
// If the resource exists already, this function will update the resource instead.
//
// Deprecated: use CreateOrUpdate() instead.
func CreateOrUpdateClusterRoleBinding(client clientset.Interface, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	return CreateOrUpdate(context.Background(), client.RbacV1().ClusterRoleBindings(), clusterRoleBinding)
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
			if apierrors.IsTimeout(err) || apierrors.IsConflict(err) || apierrors.IsServerTimeout(err) || apierrors.IsServiceUnavailable(err) {
				return false, nil
			}
			return false, *lastError
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
