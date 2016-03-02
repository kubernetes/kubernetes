/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package namespace

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedextensions "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

// contentRemainingError is used to inform the caller that content is not fully removed from the namespace
type contentRemainingError struct {
	Estimate int64
}

func (e *contentRemainingError) Error() string {
	return fmt.Sprintf("some content remains in the namespace, estimate %d seconds before it is removed", e.Estimate)
}

// updateNamespaceFunc is a function that makes an update to a namespace
type updateNamespaceFunc func(kubeClient clientset.Interface, namespace *api.Namespace) (*api.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// TODO RetryOnConflict should be a generic concept in client code
func retryOnConflictError(kubeClient clientset.Interface, namespace *api.Namespace, fn updateNamespaceFunc) (result *api.Namespace, err error) {
	latestNamespace := namespace
	for {
		result, err = fn(kubeClient, latestNamespace)
		if err == nil {
			return result, nil
		}
		if !errors.IsConflict(err) {
			return nil, err
		}
		latestNamespace, err = kubeClient.Core().Namespaces().Get(latestNamespace.Name)
		if err != nil {
			return nil, err
		}
	}
	return
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func updateNamespaceStatusFunc(kubeClient clientset.Interface, namespace *api.Namespace) (*api.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == api.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := api.Namespace{}
	newNamespace.ObjectMeta = namespace.ObjectMeta
	newNamespace.Status = namespace.Status
	newNamespace.Status.Phase = api.NamespaceTerminating
	return kubeClient.Core().Namespaces().UpdateStatus(&newNamespace)
}

// finalized returns true if the namespace.Spec.Finalizers is an empty list
func finalized(namespace *api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalizeNamespaceFunc removes the kubernetes token and finalizes the namespace
func finalizeNamespaceFunc(kubeClient clientset.Interface, namespace *api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := sets.NewString()
	for i := range namespace.Spec.Finalizers {
		if namespace.Spec.Finalizers[i] != api.FinalizerKubernetes {
			finalizerSet.Insert(string(namespace.Spec.Finalizers[i]))
		}
	}
	namespaceFinalize.Spec.Finalizers = make([]api.FinalizerName, 0, len(finalizerSet))
	for _, value := range finalizerSet.List() {
		namespaceFinalize.Spec.Finalizers = append(namespaceFinalize.Spec.Finalizers, api.FinalizerName(value))
	}
	namespace, err := kubeClient.Core().Namespaces().Finalize(&namespaceFinalize)
	if err != nil {
		// it was removed already, so life is good
		if errors.IsNotFound(err) {
			return namespace, nil
		}
	}
	return namespace, err
}

// deleteAllContent will delete all content known to the system in a namespace. It returns an estimate
// of the time remaining before the remaining resources are deleted. If estimate > 0 not all resources
// are guaranteed to be gone.
// TODO: this should use discovery to delete arbitrary namespace content
func deleteAllContent(kubeClient clientset.Interface, versions *unversioned.APIVersions, namespace string, before unversioned.Time) (estimate int64, err error) {
	err = deleteServiceAccounts(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteServices(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteReplicationControllers(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	estimate, err = deletePods(kubeClient, namespace, before)
	if err != nil {
		return estimate, err
	}
	err = deleteSecrets(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteConfigMaps(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deletePersistentVolumeClaims(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteLimitRanges(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteResourceQuotas(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	err = deleteEvents(kubeClient, namespace)
	if err != nil {
		return estimate, err
	}
	// If experimental mode, delete all experimental resources for the namespace.
	if containsVersion(versions, "extensions/v1beta1") {
		resources, err := kubeClient.Discovery().ServerResourcesForGroupVersion("extensions/v1beta1")
		if err != nil {
			return estimate, err
		}
		if containsResource(resources, "horizontalpodautoscalers") {
			err = deleteHorizontalPodAutoscalers(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
		if containsResource(resources, "ingresses") {
			err = deleteIngress(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
		if containsResource(resources, "daemonsets") {
			err = deleteDaemonSets(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
		if containsResource(resources, "jobs") {
			err = deleteJobs(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
		if containsResource(resources, "deployments") {
			err = deleteDeployments(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
		if containsResource(resources, "replicasets") {
			err = deleteReplicaSets(kubeClient.Extensions(), namespace)
			if err != nil {
				return estimate, err
			}
		}
	}
	return estimate, nil
}

// syncNamespace orchestrates deletion of a Namespace and its associated content.
func syncNamespace(kubeClient clientset.Interface, versions *unversioned.APIVersions, namespace *api.Namespace) error {
	if namespace.DeletionTimestamp == nil {
		return nil
	}

	// multiple controllers may edit a namespace during termination
	// first get the latest state of the namespace before proceeding
	// if the namespace was deleted already, don't do anything
	namespace, err := kubeClient.Core().Namespaces().Get(namespace.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	glog.V(4).Infof("Syncing namespace %s", namespace.Name)

	// ensure that the status is up to date on the namespace
	// if we get a not found error, we assume the namespace is truly gone
	namespace, err = retryOnConflictError(kubeClient, namespace, updateNamespaceStatusFunc)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// if the namespace is already finalized, delete it
	if finalized(namespace) {
		err = kubeClient.Core().Namespaces().Delete(namespace.Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		return nil
	}

	// there may still be content for us to remove
	estimate, err := deleteAllContent(kubeClient, versions, namespace.Name, *namespace.DeletionTimestamp)
	if err != nil {
		return err
	}
	if estimate > 0 {
		return &contentRemainingError{estimate}
	}

	// we have removed content, so mark it finalized by us
	result, err := retryOnConflictError(kubeClient, namespace, finalizeNamespaceFunc)
	if err != nil {
		// in normal practice, this should not be possible, but if a deployment is running
		// two controllers to do namespace deletion that share a common finalizer token it's
		// possible that a not found could occur since the other controller would have finished the delete.
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(result) {
		err = kubeClient.Core().Namespaces().Delete(namespace.Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

func deleteLimitRanges(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().LimitRanges(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteResourceQuotas(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().ResourceQuotas(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteServiceAccounts(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().ServiceAccounts(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteServices(kubeClient clientset.Interface, ns string) error {
	items, err := kubeClient.Core().Services(ns).List(api.ListOptions{})
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Core().Services(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteReplicationControllers(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().ReplicationControllers(ns).DeleteCollection(nil, api.ListOptions{})
}

func deletePods(kubeClient clientset.Interface, ns string, before unversioned.Time) (int64, error) {
	items, err := kubeClient.Core().Pods(ns).List(api.ListOptions{})
	if err != nil {
		return 0, err
	}
	expired := unversioned.Now().After(before.Time)
	var deleteOptions *api.DeleteOptions
	if expired {
		deleteOptions = api.NewDeleteOptions(0)
	}
	estimate := int64(0)
	for i := range items.Items {
		if items.Items[i].Spec.TerminationGracePeriodSeconds != nil {
			grace := *items.Items[i].Spec.TerminationGracePeriodSeconds
			if grace > estimate {
				estimate = grace
			}
		}
		err := kubeClient.Core().Pods(ns).Delete(items.Items[i].Name, deleteOptions)
		if err != nil && !errors.IsNotFound(err) {
			return 0, err
		}
	}
	if expired {
		estimate = 0
	}
	return estimate, nil
}

func deleteEvents(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().Events(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteSecrets(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().Secrets(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteConfigMaps(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().ConfigMaps(ns).DeleteCollection(nil, api.ListOptions{})
}

func deletePersistentVolumeClaims(kubeClient clientset.Interface, ns string) error {
	return kubeClient.Core().PersistentVolumeClaims(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteHorizontalPodAutoscalers(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.HorizontalPodAutoscalers(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteDaemonSets(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.DaemonSets(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteJobs(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.Jobs(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteDeployments(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.Deployments(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteReplicaSets(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.ReplicaSets(ns).DeleteCollection(nil, api.ListOptions{})
}

func deleteIngress(expClient unversionedextensions.ExtensionsInterface, ns string) error {
	return expClient.Ingresses(ns).DeleteCollection(nil, api.ListOptions{})
}

// TODO: this is duplicated logic.  Move it somewhere central?
func containsVersion(versions *unversioned.APIVersions, version string) bool {
	for ix := range versions.Versions {
		if versions.Versions[ix] == version {
			return true
		}
	}
	return false
}

// TODO: this is duplicated logic.  Move it somewhere central?
func containsResource(resources *unversioned.APIResourceList, resourceName string) bool {
	if resources == nil {
		return false
	}
	for ix := range resources.APIResources {
		resource := resources.APIResources[ix]
		if resource.Name == resourceName {
			return true
		}
	}
	return false
}
