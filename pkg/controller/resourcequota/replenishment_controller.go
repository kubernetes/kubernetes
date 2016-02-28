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

package resourcequota

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/quota/evaluator/core"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// ReplenishmentFunc is a function that is invoked when controller sees a change
// that may require a quota to be replenished (i.e. object deletion, or object moved to terminal state)
type ReplenishmentFunc func(groupKind unversioned.GroupKind, namespace string, object runtime.Object)

// ReplenishmentControllerOptions is an options struct that tells a factory
// how to configure a controller that can inform the quota system it should
// replenish quota
type ReplenishmentControllerOptions struct {
	// The kind monitored for replenishment
	GroupKind unversioned.GroupKind
	// The period that should be used to re-sync the monitored resource
	ResyncPeriod controller.ResyncPeriodFunc
	// The function to invoke when a change is observed that should trigger
	// replenishment
	ReplenishmentFunc ReplenishmentFunc
}

// PodReplenishmentUpdateFunc will replenish if the old pod was quota tracked but the new is not
func PodReplenishmentUpdateFunc(options *ReplenishmentControllerOptions) func(oldObj, newObj interface{}) {
	return func(oldObj, newObj interface{}) {
		oldPod := oldObj.(*api.Pod)
		newPod := newObj.(*api.Pod)
		if core.QuotaPod(oldPod) && !core.QuotaPod(newPod) {
			options.ReplenishmentFunc(options.GroupKind, newPod.Namespace, newPod)
		}
	}
}

// ObjectReplenenishmentDeleteFunc will replenish on every delete
func ObjectReplenishmentDeleteFunc(options *ReplenishmentControllerOptions) func(obj interface{}) {
	return func(obj interface{}) {
		metaObject, err := meta.Accessor(obj)
		if err != nil {
			tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
			if !ok {
				glog.Errorf("replenishment controller could not get object from tombstone %+v, could take up to %v before quota is replenished", obj, options.ResyncPeriod())
				utilruntime.HandleError(err)
				return
			}
			metaObject, err = meta.Accessor(tombstone.Obj)
			if err != nil {
				glog.Errorf("replenishment controller tombstone contained object that is not a meta %+v, could take up to %v before quota is replenished", tombstone.Obj, options.ResyncPeriod())
				utilruntime.HandleError(err)
				return
			}
		}
		options.ReplenishmentFunc(options.GroupKind, metaObject.GetNamespace(), nil)
	}
}

// ReplenishmentControllerFactory knows how to build replenishment controllers
type ReplenishmentControllerFactory interface {
	// NewController returns a controller configured with the specified options
	NewController(options *ReplenishmentControllerOptions) (*framework.Controller, error)
}

// replenishmentControllerFactory implements ReplenishmentControllerFactory
type replenishmentControllerFactory struct {
	kubeClient clientset.Interface
}

// NewReplenishmentControllerFactory returns a factory that knows how to build controllers
// to replenish resources when updated or deleted
func NewReplenishmentControllerFactory(kubeClient clientset.Interface) ReplenishmentControllerFactory {
	return &replenishmentControllerFactory{
		kubeClient: kubeClient,
	}
}

func (r *replenishmentControllerFactory) NewController(options *ReplenishmentControllerOptions) (*framework.Controller, error) {
	var result *framework.Controller
	switch options.GroupKind {
	case api.Kind("Pod"):
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return r.kubeClient.Core().Pods(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return r.kubeClient.Core().Pods(api.NamespaceAll).Watch(options)
				},
			},
			&api.Pod{},
			options.ResyncPeriod(),
			framework.ResourceEventHandlerFuncs{
				UpdateFunc: PodReplenishmentUpdateFunc(options),
				DeleteFunc: ObjectReplenishmentDeleteFunc(options),
			},
		)
	case api.Kind("Service"):
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return r.kubeClient.Core().Services(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return r.kubeClient.Core().Services(api.NamespaceAll).Watch(options)
				},
			},
			&api.Service{},
			options.ResyncPeriod(),
			framework.ResourceEventHandlerFuncs{
				DeleteFunc: ObjectReplenishmentDeleteFunc(options),
			},
		)
	case api.Kind("ReplicationController"):
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return r.kubeClient.Core().ReplicationControllers(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return r.kubeClient.Core().ReplicationControllers(api.NamespaceAll).Watch(options)
				},
			},
			&api.ReplicationController{},
			options.ResyncPeriod(),
			framework.ResourceEventHandlerFuncs{
				DeleteFunc: ObjectReplenishmentDeleteFunc(options),
			},
		)
	case api.Kind("PersistentVolumeClaim"):
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return r.kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return r.kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).Watch(options)
				},
			},
			&api.PersistentVolumeClaim{},
			options.ResyncPeriod(),
			framework.ResourceEventHandlerFuncs{
				DeleteFunc: ObjectReplenishmentDeleteFunc(options),
			},
		)
	case api.Kind("Secret"):
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return r.kubeClient.Core().Secrets(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return r.kubeClient.Core().Secrets(api.NamespaceAll).Watch(options)
				},
			},
			&api.PersistentVolumeClaim{},
			options.ResyncPeriod(),
			framework.ResourceEventHandlerFuncs{
				DeleteFunc: ObjectReplenishmentDeleteFunc(options),
			},
		)
	default:
		return nil, fmt.Errorf("no replenishment controller available for %s", options.GroupKind)
	}
	return result, nil
}
