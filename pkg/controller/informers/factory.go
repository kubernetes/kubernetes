/*
Copyright 2015 The Kubernetes Authors.

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

package informers

import (
	"reflect"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

// SharedInformerFactory provides interface which holds unique informers for pods, nodes, namespaces, persistent volume
// claims and persistent volumes
type SharedInformerFactory interface {
	// Start starts informers that can start AFTER the API server and controllers have started
	Start(stopCh <-chan struct{})

	Pods() PodInformer
	Nodes() NodeInformer
	Namespaces() NamespaceInformer
	PersistentVolumeClaims() PVCInformer
	PersistentVolumes() PVInformer

	DaemonSets() DaemonSetInformer
	Deployments() DeploymentInformer
	ReplicaSets() ReplicaSetInformer

	StorageClasses() StorageClassInformer
}

type sharedInformerFactory struct {
	client        clientset.Interface
	lock          sync.Mutex
	defaultResync time.Duration

	informers map[reflect.Type]cache.SharedIndexInformer
	// startedInformers is used for tracking which informers have been started
	// this allows calling of Start method multiple times
	startedInformers map[reflect.Type]bool
}

// NewSharedInformerFactory constructs a new instance of sharedInformerFactory
func NewSharedInformerFactory(client clientset.Interface, defaultResync time.Duration) SharedInformerFactory {
	return &sharedInformerFactory{
		client:           client,
		defaultResync:    defaultResync,
		informers:        make(map[reflect.Type]cache.SharedIndexInformer),
		startedInformers: make(map[reflect.Type]bool),
	}
}

// Start initializes all requested informers.
func (f *sharedInformerFactory) Start(stopCh <-chan struct{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	for informerType, informer := range f.informers {
		if !f.startedInformers[informerType] {
			go informer.Run(stopCh)
			f.startedInformers[informerType] = true
		}
	}
}

// Pods returns a SharedIndexInformer that lists and watches all pods
func (f *sharedInformerFactory) Pods() PodInformer {
	return &podInformer{sharedInformerFactory: f}
}

// Nodes returns a SharedIndexInformer that lists and watches all nodes
func (f *sharedInformerFactory) Nodes() NodeInformer {
	return &nodeInformer{sharedInformerFactory: f}
}

// Namespaces returns a SharedIndexInformer that lists and watches all namespaces
func (f *sharedInformerFactory) Namespaces() NamespaceInformer {
	return &namespaceInformer{sharedInformerFactory: f}
}

// PersistentVolumeClaims returns a SharedIndexInformer that lists and watches all persistent volume claims
func (f *sharedInformerFactory) PersistentVolumeClaims() PVCInformer {
	return &pvcInformer{sharedInformerFactory: f}
}

// PersistentVolumes returns a SharedIndexInformer that lists and watches all persistent volumes
func (f *sharedInformerFactory) PersistentVolumes() PVInformer {
	return &pvInformer{sharedInformerFactory: f}
}

func (f *sharedInformerFactory) DaemonSets() DaemonSetInformer {
	return &daemonSetInformer{sharedInformerFactory: f}
}

func (f *sharedInformerFactory) Deployments() DeploymentInformer {
	return &deploymentInformer{sharedInformerFactory: f}
}

func (f *sharedInformerFactory) ReplicaSets() ReplicaSetInformer {
	return &replicaSetInformer{sharedInformerFactory: f}
}

// StorageClasses returns a SharedIndexInformer that lists and watches all storage classes
func (f *sharedInformerFactory) StorageClasses() StorageClassInformer {
	return &storageClassInformer{sharedInformerFactory: f}
}
