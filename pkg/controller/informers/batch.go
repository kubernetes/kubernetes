/*
Copyright 2016 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	batchv1listers "k8s.io/kubernetes/pkg/client/listers/batch/v1"
)

// JobInformer is type of SharedIndexInformer which watches and lists all jobs.
// Interface provides constructor for informer and lister for jobs
type JobInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() batchv1listers.JobLister
}

type jobInformer struct {
	*sharedInformerFactory
}

// Informer checks whether jobInformer exists in sharedInformerFactory and if not, it creates new informer of type
// jobInformer and connects it to sharedInformerFactory
func (f *jobInformer) Informer() cache.SharedIndexInformer {
	f.lock.Lock()
	defer f.lock.Unlock()

	informerType := reflect.TypeOf(&batch.Job{})
	informer, exists := f.informers[informerType]
	if exists {
		return informer
	}
	informer = NewJobInformer(f.client, f.defaultResync)
	f.informers[informerType] = informer

	return informer
}

// NewJobInformer returns a SharedIndexInformer that lists and watches all jobs
func NewJobInformer(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	sharedIndexInformer := cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return client.Batch().Jobs(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return client.Batch().Jobs(metav1.NamespaceAll).Watch(options)
			},
		},
		&batch.Job{},
		resyncPeriod,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	return sharedIndexInformer
}

// Lister returns lister for jobInformer
func (f *jobInformer) Lister() batchv1listers.JobLister {
	informer := f.Informer()
	return batchv1listers.NewJobLister(informer.GetIndexer())
}
