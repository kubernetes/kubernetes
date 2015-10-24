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

package namespacecontroller

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// NamespaceController is responsible for performing actions dependent upon a namespace phase
type NamespaceController struct {
	controller     *framework.Controller
	StopEverything chan struct{}
}

// NewNamespaceController creates a new NamespaceController
func NewNamespaceController(kubeClient client.Interface, versions *unversioned.APIVersions, resyncPeriod time.Duration) *NamespaceController {
	var controller *framework.Controller
	_, controller = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Namespaces().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Namespaces().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.Namespace{},
		// TODO: Can we have much longer period here?
		resyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				namespace := obj.(*api.Namespace)
				if err := syncNamespace(kubeClient, versions, namespace); err != nil {
					if estimate, ok := err.(*contentRemainingError); ok {
						go func() {
							// Estimate is the aggregate total of TerminationGracePeriodSeconds, which defaults to 30s
							// for pods.  However, most processes will terminate faster - within a few seconds, probably
							// with a peak within 5-10s.  So this division is a heuristic that avoids waiting the full
							// duration when in many cases things complete more quickly. The extra second added is to
							// ensure we never wait 0 seconds.
							t := estimate.Estimate/2 + 1
							glog.V(4).Infof("Content remaining in namespace %s, waiting %d seconds", namespace.Name, t)
							time.Sleep(time.Duration(t) * time.Second)
							if err := controller.Requeue(namespace); err != nil {
								util.HandleError(err)
							}
						}()
						return
					}
					util.HandleError(err)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				namespace := newObj.(*api.Namespace)
				if err := syncNamespace(kubeClient, versions, namespace); err != nil {
					if estimate, ok := err.(*contentRemainingError); ok {
						go func() {
							t := estimate.Estimate/2 + 1
							glog.V(4).Infof("Content remaining in namespace %s, waiting %d seconds", namespace.Name, t)
							time.Sleep(time.Duration(t) * time.Second)
							if err := controller.Requeue(namespace); err != nil {
								util.HandleError(err)
							}
						}()
						return
					}
					util.HandleError(err)
				}
			},
		},
	)

	return &NamespaceController{
		controller: controller,
	}
}

// Run begins observing the system.  It starts a goroutine and returns immediately.
func (nm *NamespaceController) Run() {
	if nm.StopEverything == nil {
		nm.StopEverything = make(chan struct{})
		go nm.controller.Run(nm.StopEverything)
	}
}

// Stop gracefully shutsdown this controller
func (nm *NamespaceController) Stop() {
	if nm.StopEverything != nil {
		close(nm.StopEverything)
		nm.StopEverything = nil
	}
}

// finalized returns true if the spec.finalizers is empty list
func finalized(namespace *api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalize will finalize the namespace for kubernetes
func finalizeNamespaceFunc(kubeClient client.Interface, namespace *api.Namespace) (*api.Namespace, error) {
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
	return kubeClient.Namespaces().Finalize(&namespaceFinalize)
}

type contentRemainingError struct {
	Estimate int64
}

func (e *contentRemainingError) Error() string {
	return fmt.Sprintf("some content remains in the namespace, estimate %d seconds before it is removed", e.Estimate)
}

// deleteAllContent will delete all content known to the system in a namespace. It returns an estimate
// of the time remaining before the remaining resources are deleted. If estimate > 0 not all resources
// are guaranteed to be gone.
func deleteAllContent(kubeClient client.Interface, versions *unversioned.APIVersions, namespace string, before unversioned.Time) (estimate int64, err error) {
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
		resources, err := kubeClient.SupportedResourcesForGroupVersion("extensions/v1beta1")
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
	}
	return estimate, nil
}

// updateNamespaceFunc is a function that makes an update to a namespace
type updateNamespaceFunc func(kubeClient client.Interface, namespace *api.Namespace) (*api.Namespace, error)

// retryOnConflictError retries the specified fn if there was a conflict error
// TODO RetryOnConflict should be a generic concept in client code
func retryOnConflictError(kubeClient client.Interface, namespace *api.Namespace, fn updateNamespaceFunc) (result *api.Namespace, err error) {
	latestNamespace := namespace
	for {
		result, err = fn(kubeClient, latestNamespace)
		if err == nil {
			return result, nil
		}
		if !errors.IsConflict(err) {
			return nil, err
		}
		latestNamespace, err = kubeClient.Namespaces().Get(latestNamespace.Name)
		if err != nil {
			return nil, err
		}
	}
	return
}

// updateNamespaceStatusFunc will verify that the status of the namespace is correct
func updateNamespaceStatusFunc(kubeClient client.Interface, namespace *api.Namespace) (*api.Namespace, error) {
	if namespace.DeletionTimestamp.IsZero() || namespace.Status.Phase == api.NamespaceTerminating {
		return namespace, nil
	}
	newNamespace := api.Namespace{}
	newNamespace.ObjectMeta = namespace.ObjectMeta
	newNamespace.Status = namespace.Status
	newNamespace.Status.Phase = api.NamespaceTerminating
	return kubeClient.Namespaces().Status(&newNamespace)
}

// syncNamespace orchestrates deletion of a Namespace and its associated content.
func syncNamespace(kubeClient client.Interface, versions *unversioned.APIVersions, namespace *api.Namespace) (err error) {
	if namespace.DeletionTimestamp == nil {
		return nil
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
		err = kubeClient.Namespaces().Delete(namespace.Name)
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
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(result) {
		err = kubeClient.Namespaces().Delete(namespace.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

func deleteLimitRanges(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.LimitRanges(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.LimitRanges(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteResourceQuotas(kubeClient client.Interface, ns string) error {
	resourceQuotas, err := kubeClient.ResourceQuotas(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range resourceQuotas.Items {
		err := kubeClient.ResourceQuotas(ns).Delete(resourceQuotas.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteServiceAccounts(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.ServiceAccounts(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.ServiceAccounts(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteServices(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Services(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Services(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteReplicationControllers(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.ReplicationControllers(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.ReplicationControllers(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deletePods(kubeClient client.Interface, ns string, before unversioned.Time) (int64, error) {
	items, err := kubeClient.Pods(ns).List(labels.Everything(), fields.Everything())
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
		err := kubeClient.Pods(ns).Delete(items.Items[i].Name, deleteOptions)
		if err != nil && !errors.IsNotFound(err) {
			return 0, err
		}
	}
	if expired {
		estimate = 0
	}
	return estimate, nil
}

func deleteEvents(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Events(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Events(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteSecrets(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.Secrets(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.Secrets(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deletePersistentVolumeClaims(kubeClient client.Interface, ns string) error {
	items, err := kubeClient.PersistentVolumeClaims(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := kubeClient.PersistentVolumeClaims(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteHorizontalPodAutoscalers(expClient client.ExtensionsInterface, ns string) error {
	items, err := expClient.HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := expClient.HorizontalPodAutoscalers(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteDaemonSets(expClient client.ExtensionsInterface, ns string) error {
	items, err := expClient.DaemonSets(ns).List(labels.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := expClient.DaemonSets(ns).Delete(items.Items[i].Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteJobs(expClient client.ExtensionsInterface, ns string) error {
	items, err := expClient.Jobs(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := expClient.Jobs(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteDeployments(expClient client.ExtensionsInterface, ns string) error {
	items, err := expClient.Deployments(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := expClient.Deployments(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func deleteIngress(expClient client.ExtensionsInterface, ns string) error {
	items, err := expClient.Ingress(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range items.Items {
		err := expClient.Ingress(ns).Delete(items.Items[i].Name, nil)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}
	return nil
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
