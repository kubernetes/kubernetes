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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

type monitoringControllerFactory struct {
	kubeClient client.Interface
}

func (f *monitoringControllerFactory) NewController(options *MonitoringControllerOptions) (*framework.Controller, error) {
	var result *framework.Controller

	switch options.GroupKind {
	case unversioned.GroupKind{Group: "", Kind: "Pod"}:
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return f.kubeClient.Pods(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return f.kubeClient.Pods(api.NamespaceAll).Watch(options)
				},
			},
			&api.Pod{},
			options.ResyncPeriod(),
			options.ResourceEventHandlerFuncs,
		)
	case unversioned.GroupKind{Group: "", Kind: "Service"}:
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return f.kubeClient.Services(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return f.kubeClient.Services(api.NamespaceAll).Watch(options)
				},
			},
			&api.Service{},
			options.ResyncPeriod(),
			options.ResourceEventHandlerFuncs,
		)
	case unversioned.GroupKind{Group: "", Kind: "ReplicationController"}:
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return f.kubeClient.ReplicationControllers(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return f.kubeClient.ReplicationControllers(api.NamespaceAll).Watch(options)
				},
			},
			&api.ReplicationController{},
			options.ResyncPeriod(),
			options.ResourceEventHandlerFuncs,
		)
	case unversioned.GroupKind{Group: "", Kind: "PersistentVolumeClaim"}:
		_, result = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return f.kubeClient.PersistentVolumeClaims(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return f.kubeClient.PersistentVolumeClaims(api.NamespaceAll).Watch(options)
				},
			},
			&api.PersistentVolumeClaim{},
			options.ResyncPeriod(),
			options.ResourceEventHandlerFuncs,
		)
	default:
		return nil, fmt.Errorf("No monitoring controller defined for %s", options.GroupKind)
	}

	return result, nil
}

// NewMonitoringControllerFactory returns factory that supports monitoring core kubernetes resources
func NewMonitoringControllerFactory(kubeClient client.Interface) MonitoringControllerFactory {
	return &monitoringControllerFactory{kubeClient: kubeClient}
}
