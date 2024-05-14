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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// ConfigMapMutator is a function that mutates the given ConfigMap and optionally returns an error
type ConfigMapMutator func(*v1.ConfigMap) error

// apiCallRetryInterval holds a local copy of apiCallRetryInterval for testing purposes
var apiCallRetryInterval = constants.KubernetesAPICallRetryInterval

// TODO: We should invent a dynamic mechanism for this using the dynamic client instead of hard-coding these functions per-type

// CreateOrUpdateConfigMap creates a ConfigMap if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateConfigMap(client clientset.Interface, cm *v1.ConfigMap) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create ConfigMap")
					return false, nil
				}
				if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Update(ctx, cm, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update ConfigMap")
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

// CreateOrMutateConfigMap tries to create the ConfigMap provided as cm. If the resource exists already, the latest version will be fetched from
// the cluster and mutator callback will be called on it, then an Update of the mutated ConfigMap will be performed. This function is resilient
// to conflicts, and a retry will be issued if the ConfigMap was modified on the server between the refresh and the update (while the mutation was
// taking place)
func CreateOrMutateConfigMap(client clientset.Interface, cm *v1.ConfigMap, mutator ConfigMapMutator) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(context.Background(), cm, metav1.CreateOptions{}); err != nil {
				lastError = err
				if apierrors.IsAlreadyExists(err) {
					lastError = mutateConfigMap(client, metav1.ObjectMeta{Namespace: cm.ObjectMeta.Namespace, Name: cm.ObjectMeta.Name}, mutator)
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

// mutateConfigMap takes a ConfigMap Object Meta (namespace and name), retrieves the resource from the server and tries to mutate it
// by calling to the mutator callback, then an Update of the mutated ConfigMap will be performed. This function is resilient
// to conflicts, and a retry will be issued if the ConfigMap was modified on the server between the refresh and the update (while the mutation was
// taking place).
func mutateConfigMap(client clientset.Interface, meta metav1.ObjectMeta, mutator ConfigMapMutator) error {
	ctx := context.Background()
	configMap, err := client.CoreV1().ConfigMaps(meta.Namespace).Get(ctx, meta.Name, metav1.GetOptions{})
	if err != nil {
		return errors.Wrap(err, "unable to get ConfigMap")
	}
	if err = mutator(configMap); err != nil {
		return errors.Wrap(err, "unable to mutate ConfigMap")
	}
	_, err = client.CoreV1().ConfigMaps(configMap.ObjectMeta.Namespace).Update(ctx, configMap, metav1.UpdateOptions{})
	return err
}

// CreateOrRetainConfigMap creates a ConfigMap if the target resource doesn't exist. If the resource exists already, this function will retain the resource instead.
func CreateOrRetainConfigMap(client clientset.Interface, cm *v1.ConfigMap, configMapName string) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Get(ctx, configMapName, metav1.GetOptions{}); err != nil {
				if !apierrors.IsNotFound(err) {
					lastError = errors.Wrap(err, "unable to get ConfigMap")
					return false, nil
				}
				if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to create ConfigMap")
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

// CreateOrUpdateSecret creates a Secret if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateSecret(client clientset.Interface, secret *v1.Secret) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.CoreV1().Secrets(secret.ObjectMeta.Namespace).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create Secret")
					return false, nil
				}
				if _, err := client.CoreV1().Secrets(secret.ObjectMeta.Namespace).Update(ctx, secret, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update Secret")
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

// CreateOrUpdateServiceAccount creates a ServiceAccount if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateServiceAccount(client clientset.Interface, sa *v1.ServiceAccount) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.CoreV1().ServiceAccounts(sa.ObjectMeta.Namespace).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create ServicAccount")
					return false, nil
				}
				if _, err := client.CoreV1().ServiceAccounts(sa.ObjectMeta.Namespace).Update(ctx, sa, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update ServicAccount")
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

// CreateOrUpdateDeployment creates a Deployment if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateDeployment(client clientset.Interface, deploy *apps.Deployment) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Create(ctx, deploy, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create Deployment")
					return false, nil
				}
				if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Update(ctx, deploy, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update Deployment")
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

// CreateOrRetainDeployment creates a Deployment if the target resource doesn't exist. If the resource exists already, this function will retain the resource instead.
func CreateOrRetainDeployment(client clientset.Interface, deploy *apps.Deployment, deployName string) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Get(ctx, deployName, metav1.GetOptions{}); err != nil {
				if !apierrors.IsNotFound(err) {
					lastError = errors.Wrap(err, "unable to get Deployment")
					return false, nil
				}
				if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Create(ctx, deploy, metav1.CreateOptions{}); err != nil {
					if !apierrors.IsAlreadyExists(err) {
						lastError = errors.Wrap(err, "unable to create Deployment")
						return false, nil
					}
				}
			}
			return true, nil
		})
	if err == nil {
		return nil
	}
	return lastError
}

// CreateOrUpdateDaemonSet creates a DaemonSet if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateDaemonSet(client clientset.Interface, ds *apps.DaemonSet) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.AppsV1().DaemonSets(ds.ObjectMeta.Namespace).Create(ctx, ds, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create DaemonSet")
					return false, nil
				}
				if _, err := client.AppsV1().DaemonSets(ds.ObjectMeta.Namespace).Update(ctx, ds, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update DaemonSet")
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

// CreateOrUpdateRole creates a Role if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateRole(client clientset.Interface, role *rbac.Role) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.RbacV1().Roles(role.ObjectMeta.Namespace).Create(ctx, role, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create Role")
					return false, nil
				}
				if _, err := client.RbacV1().Roles(role.ObjectMeta.Namespace).Update(ctx, role, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update Role")
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

// CreateOrUpdateRoleBinding creates a RoleBinding if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateRoleBinding(client clientset.Interface, roleBinding *rbac.RoleBinding) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.RbacV1().RoleBindings(roleBinding.ObjectMeta.Namespace).Create(ctx, roleBinding, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create RoleBinding")
					return false, nil
				}
				if _, err := client.RbacV1().RoleBindings(roleBinding.ObjectMeta.Namespace).Update(ctx, roleBinding, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update RoleBinding")
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

// CreateOrUpdateClusterRole creates a ClusterRole if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateClusterRole(client clientset.Interface, clusterRole *rbac.ClusterRole) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.RbacV1().ClusterRoles().Create(ctx, clusterRole, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create ClusterRole")
					return false, nil
				}
				if _, err := client.RbacV1().ClusterRoles().Update(ctx, clusterRole, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update ClusterRole")
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

// CreateOrUpdateClusterRoleBinding creates a ClusterRoleBinding if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateClusterRoleBinding(client clientset.Interface, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	var lastError error
	err := wait.PollUntilContextTimeout(context.Background(),
		apiCallRetryInterval, kubeadmapi.GetActiveTimeouts().KubernetesAPICall.Duration,
		true, func(_ context.Context) (bool, error) {
			ctx := context.Background()
			if _, err := client.RbacV1().ClusterRoleBindings().Create(ctx, clusterRoleBinding, metav1.CreateOptions{}); err != nil {
				if !apierrors.IsAlreadyExists(err) {
					lastError = errors.Wrap(err, "unable to create ClusterRoleBinding")
					return false, nil
				}
				if _, err := client.RbacV1().ClusterRoleBindings().Update(ctx, clusterRoleBinding, metav1.UpdateOptions{}); err != nil {
					lastError = errors.Wrap(err, "unable to update ClusterRoleBinding")
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
