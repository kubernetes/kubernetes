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
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// NamespaceController is responsible for performing actions dependent upon a namespace phase
type NamespaceController struct {
	controller     *framework.Controller
	StopEverything chan struct{}
}

// NewNamespaceController creates a new NamespaceController
func NewNamespaceController(kubeClient client.Interface, resyncPeriod time.Duration) *NamespaceController {
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
		resyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				namespace := obj.(*api.Namespace)
				if err := syncNamespace(kubeClient, *namespace); err != nil {
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
				if err := syncNamespace(kubeClient, *namespace); err != nil {
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
func finalized(namespace api.Namespace) bool {
	return len(namespace.Spec.Finalizers) == 0
}

// finalize will finalize the namespace for kubernetes
func finalize(kubeClient client.Interface, namespace api.Namespace) (*api.Namespace, error) {
	namespaceFinalize := api.Namespace{}
	namespaceFinalize.ObjectMeta = namespace.ObjectMeta
	namespaceFinalize.Spec = namespace.Spec
	finalizerSet := util.NewStringSet()
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
func deleteAllContent(kubeClient client.Interface, namespace string, before util.Time) (estimate int64, err error) {
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
	err = deletePods(kubeClient, namespace)
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

	return estimate, nil
}

// syncNamespace makes namespace life-cycle decisions
func syncNamespace(kubeClient client.Interface, namespace api.Namespace) (err error) {
	if namespace.DeletionTimestamp == nil {
		return nil
	}
	glog.V(4).Infof("Syncing namespace %s", namespace.Name)

	// if there is a deletion timestamp, and the status is not terminating, then update status
	if !namespace.DeletionTimestamp.IsZero() && namespace.Status.Phase != api.NamespaceTerminating {
		newNamespace := api.Namespace{}
		newNamespace.ObjectMeta = namespace.ObjectMeta
		newNamespace.Status = namespace.Status
		newNamespace.Status.Phase = api.NamespaceTerminating
		result, err := kubeClient.Namespaces().Status(&newNamespace)
		if err != nil {
			return err
		}
		// work with the latest copy so we can proceed to clean up right away without another interval
		namespace = *result
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
	estimate, err := deleteAllContent(kubeClient, namespace.Name, *namespace.DeletionTimestamp)
	if err != nil {
		return err
	}
	if estimate > 0 {
		return &contentRemainingError{estimate}
	}

	// we have removed content, so mark it finalized by us
	result, err := finalize(kubeClient, namespace)
	if err != nil {
		return err
	}

	// now check if all finalizers have reported that we delete now
	if finalized(*result) {
		err = kubeClient.Namespaces().Delete(namespace.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
	}

	return nil
}

func deleteLimitRanges(kubeClient client.Interface, ns string) error {
	if err := kubeClient.LimitRanges(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteResourceQuotas(kubeClient client.Interface, ns string) error {
	if err := kubeClient.ResourceQuotas(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteServiceAccounts(kubeClient client.Interface, ns string) error {
	if err := kubeClient.ServiceAccounts(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteServices(kubeClient client.Interface, ns string) error {
	if err := kubeClient.Services(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteReplicationControllers(kubeClient client.Interface, ns string) error {
	if err := kubeClient.ReplicationControllers(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deletePods(kubeClient client.Interface, ns string) error {
	if err := kubeClient.Pods(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteEvents(kubeClient client.Interface, ns string) error {
	if err := kubeClient.Events(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deleteSecrets(kubeClient client.Interface, ns string) error {
	if err := kubeClient.Secrets(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}

func deletePersistentVolumeClaims(kubeClient client.Interface, ns string) error {
	if err := kubeClient.PersistentVolumeClaims(ns).DeleteAll(); err != nil {
		return err
	}
	return nil
}
