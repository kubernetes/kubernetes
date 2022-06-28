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
	"fmt"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clientsetretry "k8s.io/client-go/util/retry"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// ConfigMapMutator is a function that mutates the given ConfigMap and optionally returns an error
type ConfigMapMutator func(*v1.ConfigMap) error

// TODO: We should invent a dynamic mechanism for this using the dynamic client instead of hard-coding these functions per-type

// CreateOrUpdateConfigMap creates a ConfigMap if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateConfigMap(client clientset.Interface, cm *v1.ConfigMap) error {
	if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create ConfigMap: %w", err)
		}

		if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Update(context.TODO(), cm, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update ConfigMap: %w", err)
		}
	}
	return nil
}

// CreateOrMutateConfigMap tries to create the ConfigMap provided as cm. If the resource exists already, the latest version will be fetched from
// the cluster and mutator callback will be called on it, then an Update of the mutated ConfigMap will be performed. This function is resilient
// to conflicts, and a retry will be issued if the ConfigMap was modified on the server between the refresh and the update (while the mutation was
// taking place)
func CreateOrMutateConfigMap(client clientset.Interface, cm *v1.ConfigMap, mutator ConfigMapMutator) error {
	var lastError error
	err := wait.PollImmediate(constants.APICallRetryInterval, constants.APICallWithWriteTimeout, func() (bool, error) {
		if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{}); err != nil {
			lastError = err
			if apierrors.IsAlreadyExists(err) {
				lastError = MutateConfigMap(client, metav1.ObjectMeta{Namespace: cm.ObjectMeta.Namespace, Name: cm.ObjectMeta.Name}, mutator)
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

// MutateConfigMap takes a ConfigMap Object Meta (namespace and name), retrieves the resource from the server and tries to mutate it
// by calling to the mutator callback, then an Update of the mutated ConfigMap will be performed. This function is resilient
// to conflicts, and a retry will be issued if the ConfigMap was modified on the server between the refresh and the update (while the mutation was
// taking place).
func MutateConfigMap(client clientset.Interface, meta metav1.ObjectMeta, mutator ConfigMapMutator) error {
	var lastError error
	err := wait.PollImmediate(constants.APICallRetryInterval, constants.APICallWithWriteTimeout, func() (bool, error) {
		configMap, err := client.CoreV1().ConfigMaps(meta.Namespace).Get(context.TODO(), meta.Name, metav1.GetOptions{})
		if err != nil {
			lastError = err
			return false, nil
		}
		if err = mutator(configMap); err != nil {
			lastError = fmt.Errorf("unable to mutate ConfigMap: %w", err)
			return false, nil
		}
		_, lastError = client.CoreV1().ConfigMaps(configMap.ObjectMeta.Namespace).Update(context.TODO(), configMap, metav1.UpdateOptions{})
		return lastError == nil, nil
	})
	if err == nil {
		return nil
	}
	return lastError
}

// CreateOrRetainConfigMap creates a ConfigMap if the target resource doesn't exist. If the resource exists already, this function will retain the resource instead.
func CreateOrRetainConfigMap(client clientset.Interface, cm *v1.ConfigMap, configMapName string) error {
	if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Get(context.TODO(), configMapName, metav1.GetOptions{}); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil
		}
		if _, err := client.CoreV1().ConfigMaps(cm.ObjectMeta.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{}); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("unable to create ConfigMap: %w", err)
			}
		}
	}
	return nil
}

// CreateOrUpdateSecret creates a Secret if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateSecret(client clientset.Interface, secret *v1.Secret) error {
	if _, err := client.CoreV1().Secrets(secret.ObjectMeta.Namespace).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create secret: %w", err)
		}

		if _, err := client.CoreV1().Secrets(secret.ObjectMeta.Namespace).Update(context.TODO(), secret, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update secret: %w", err)
		}
	}
	return nil
}

// CreateOrUpdateServiceAccount creates a ServiceAccount if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateServiceAccount(client clientset.Interface, sa *v1.ServiceAccount) error {
	if _, err := client.CoreV1().ServiceAccounts(sa.ObjectMeta.Namespace).Create(context.TODO(), sa, metav1.CreateOptions{}); err != nil {
		// Note: We don't run .Update here afterwards as that's probably not required
		// Only thing that could be updated is annotations/labels in .metadata, but we don't use that currently
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create serviceaccount: %w", err)
		}
	}
	return nil
}

// CreateOrUpdateDeployment creates a Deployment if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateDeployment(client clientset.Interface, deploy *apps.Deployment) error {
	if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Create(context.TODO(), deploy, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create deployment: %w", err)
		}

		if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Update(context.TODO(), deploy, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update deployment: %w", err)
		}
	}
	return nil
}

// CreateOrRetainDeployment creates a Deployment if the target resource doesn't exist. If the resource exists already, this function will retain the resource instead.
func CreateOrRetainDeployment(client clientset.Interface, deploy *apps.Deployment, deployName string) error {
	if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Get(context.TODO(), deployName, metav1.GetOptions{}); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil
		}
		if _, err := client.AppsV1().Deployments(deploy.ObjectMeta.Namespace).Create(context.TODO(), deploy, metav1.CreateOptions{}); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("unable to create deployment: %w", err)
			}
		}
	}
	return nil
}

// CreateOrUpdateDaemonSet creates a DaemonSet if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateDaemonSet(client clientset.Interface, ds *apps.DaemonSet) error {
	if _, err := client.AppsV1().DaemonSets(ds.ObjectMeta.Namespace).Create(context.TODO(), ds, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create daemonset: %w", err)
		}

		if _, err := client.AppsV1().DaemonSets(ds.ObjectMeta.Namespace).Update(context.TODO(), ds, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update daemonset: %w", err)
		}
	}
	return nil
}

// DeleteDaemonSetForeground deletes the specified DaemonSet in foreground mode; i.e. it blocks until/makes sure all the managed Pods are deleted
func DeleteDaemonSetForeground(client clientset.Interface, namespace, name string) error {
	foregroundDelete := metav1.DeletePropagationForeground
	return client.AppsV1().DaemonSets(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &foregroundDelete})
}

// DeleteDeploymentForeground deletes the specified Deployment in foreground mode; i.e. it blocks until/makes sure all the managed Pods are deleted
func DeleteDeploymentForeground(client clientset.Interface, namespace, name string) error {
	foregroundDelete := metav1.DeletePropagationForeground
	return client.AppsV1().Deployments(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &foregroundDelete})
}

// CreateOrUpdateRole creates a Role if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateRole(client clientset.Interface, role *rbac.Role) error {
	var lastError error
	err := wait.PollImmediate(constants.APICallRetryInterval, constants.APICallWithWriteTimeout, func() (bool, error) {
		if _, err := client.RbacV1().Roles(role.ObjectMeta.Namespace).Create(context.TODO(), role, metav1.CreateOptions{}); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				lastError = fmt.Errorf("unable to create RBAC role: %w", err)
				return false, nil
			}

			if _, err := client.RbacV1().Roles(role.ObjectMeta.Namespace).Update(context.TODO(), role, metav1.UpdateOptions{}); err != nil {
				lastError = fmt.Errorf("unable to update RBAC role: %w", err)
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
	err := wait.PollImmediate(constants.APICallRetryInterval, constants.APICallWithWriteTimeout, func() (bool, error) {
		if _, err := client.RbacV1().RoleBindings(roleBinding.ObjectMeta.Namespace).Create(context.TODO(), roleBinding, metav1.CreateOptions{}); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				lastError = fmt.Errorf("unable to create RBAC rolebinding: %w", err)
				return false, nil
			}

			if _, err := client.RbacV1().RoleBindings(roleBinding.ObjectMeta.Namespace).Update(context.TODO(), roleBinding, metav1.UpdateOptions{}); err != nil {
				lastError = fmt.Errorf("unable to update RBAC rolebinding: %w", err)
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
	if _, err := client.RbacV1().ClusterRoles().Create(context.TODO(), clusterRole, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC clusterrole: %w", err)
		}

		if _, err := client.RbacV1().ClusterRoles().Update(context.TODO(), clusterRole, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update RBAC clusterrole: %w", err)
		}
	}
	return nil
}

// CreateOrUpdateClusterRoleBinding creates a ClusterRoleBinding if the target resource doesn't exist. If the resource exists already, this function will update the resource instead.
func CreateOrUpdateClusterRoleBinding(client clientset.Interface, clusterRoleBinding *rbac.ClusterRoleBinding) error {
	if _, err := client.RbacV1().ClusterRoleBindings().Create(context.TODO(), clusterRoleBinding, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("unable to create RBAC clusterrolebinding: %w", err)
		}

		if _, err := client.RbacV1().ClusterRoleBindings().Update(context.TODO(), clusterRoleBinding, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("unable to update RBAC clusterrolebinding: %w", err)
		}
	}
	return nil
}

// PatchNodeOnce executes patchFn on the node object found by the node name.
// This is a condition function meant to be used with wait.Poll. false, nil
// implies it is safe to try again, an error indicates no more tries should be
// made and true indicates success.
func PatchNodeOnce(client clientset.Interface, nodeName string, patchFn func(*v1.Node), lastError *error) func() (bool, error) {
	return func() (bool, error) {
		// First get the node object
		n, err := client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
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
			*lastError = fmt.Errorf("failed to marshal unmodified node %q into JSON: %w", n.Name, err)
			return false, *lastError
		}

		// Execute the mutating function
		patchFn(n)

		newData, err := json.Marshal(n)
		if err != nil {
			*lastError = fmt.Errorf("failed to marshal modified node %q into JSON: %w", n.Name, err)
			return false, *lastError
		}

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		if err != nil {
			*lastError = fmt.Errorf("failed to create two way merge patch: %w", err)
			return false, *lastError
		}

		if _, err := client.CoreV1().Nodes().Patch(context.TODO(), n.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}); err != nil {
			*lastError = fmt.Errorf("error patching node %q through apiserver: %w", n.Name, err)
			if apierrors.IsTimeout(err) || apierrors.IsConflict(err) {
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
	// wait.Poll will rerun the condition function every interval function if
	// the function returns false. If the condition function returns an error
	// then the retries end and the error is returned.
	err := wait.Poll(constants.APICallRetryInterval, constants.PatchNodeTimeout, PatchNodeOnce(client, nodeName, patchFn, &lastError))
	if err == nil {
		return nil
	}
	return lastError
}

// GetConfigMapWithRetry tries to retrieve a ConfigMap using the given client,
// retrying if we get an unexpected error.
//
// TODO: evaluate if this can be done better. Potentially remove the retry if feasible.
func GetConfigMapWithRetry(client clientset.Interface, namespace, name string) (*v1.ConfigMap, error) {
	var cm *v1.ConfigMap
	var lastError error
	err := wait.ExponentialBackoff(clientsetretry.DefaultBackoff, func() (bool, error) {
		var err error
		cm, err = client.CoreV1().ConfigMaps(namespace).Get(context.TODO(), name, metav1.GetOptions{})
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
