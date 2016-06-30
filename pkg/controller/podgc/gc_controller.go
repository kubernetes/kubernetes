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

package podgc

import (
	"sort"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	gcCheckPeriod = 20 * time.Second
)

type PodGCController struct {
	kubeClient     clientset.Interface
	podStore       cache.StoreToPodLister
	podStoreSyncer *framework.Controller
	deletePod      func(namespace, name string) error
	threshold      int
}

func New(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, threshold int) *PodGCController {
	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("gc_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}
	gcc := &PodGCController{
		kubeClient: kubeClient,
		threshold:  threshold,
		deletePod: func(namespace, name string) error {
			return kubeClient.Core().Pods(namespace).Delete(name, api.NewDeleteOptions(0))
		},
	}

	terminatedSelector := fields.ParseSelectorOrDie("status.phase!=" + string(api.PodPending) + ",status.phase!=" + string(api.PodRunning) + ",status.phase!=" + string(api.PodUnknown))

	gcc.podStore.Indexer, gcc.podStoreSyncer = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.FieldSelector = terminatedSelector
				return gcc.kubeClient.Core().Pods(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.FieldSelector = terminatedSelector
				return gcc.kubeClient.Core().Pods(api.NamespaceAll).Watch(options)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{},
		// We don't need to build a index for podStore here actually, but build one for consistency.
		// It will ensure that if people start making use of the podStore in more specific ways,
		// they'll get the benefits they expect. It will also reserve the name for future refactorings.
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)
	return gcc
}

func (gcc *PodGCController) Run(stop <-chan struct{}) {
	go gcc.podStoreSyncer.Run(stop)
	go wait.Until(gcc.gc, gcCheckPeriod, stop)
	<-stop
}

func (gcc *PodGCController) gc() {
	terminatedPods, _ := gcc.podStore.List(labels.Everything())
	terminatedPodCount := len(terminatedPods)
	sort.Sort(byCreationTimestamp(terminatedPods))

	deleteCount := terminatedPodCount - gcc.threshold

	if deleteCount > terminatedPodCount {
		deleteCount = terminatedPodCount
	}
	if deleteCount > 0 {
		glog.Infof("garbage collecting %v pods", deleteCount)
	}

	var wait sync.WaitGroup
	for i := 0; i < deleteCount; i++ {
		wait.Add(1)
		go func(namespace string, name string) {
			defer wait.Done()
			if err := gcc.deletePod(namespace, name); err != nil {
				// ignore not founds
				defer utilruntime.HandleError(err)
			}
		}(terminatedPods[i].Namespace, terminatedPods[i].Name)
	}
	wait.Wait()
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []*api.Pod

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
