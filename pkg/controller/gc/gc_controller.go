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

package gc

import (
	"sort"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	gcCheckPeriod = 20 * time.Second
)

type GCController struct {
	kubeClient     client.Interface
	podStore       cache.StoreToPodLister
	podStoreSyncer *framework.Controller
	deletePod      func(namespace, name string) error
	threshold      int
}

func New(kubeClient client.Interface, resyncPeriod controller.ResyncPeriodFunc, threshold int) *GCController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	gcc := &GCController{
		kubeClient: kubeClient,
		threshold:  threshold,
		deletePod: func(namespace, name string) error {
			return kubeClient.Pods(namespace).Delete(name, api.NewDeleteOptions(0))
		},
	}

	terminatedSelector := compileTerminatedPodSelector()

	gcc.podStore.Store, gcc.podStoreSyncer = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return gcc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), terminatedSelector)
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return gcc.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), terminatedSelector, rv)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{},
	)
	return gcc
}

func (gcc *GCController) Run(stop <-chan struct{}) {
	go gcc.podStoreSyncer.Run(stop)
	go util.Until(gcc.gc, gcCheckPeriod, stop)
	<-stop
}

func (gcc *GCController) gc() {
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
				defer util.HandleError(err)
			}
		}(terminatedPods[i].Namespace, terminatedPods[i].Name)
	}
	wait.Wait()
}

func compileTerminatedPodSelector() fields.Selector {
	selector, err := fields.ParseSelector("status.phase!=" + string(api.PodPending) + ",status.phase!=" + string(api.PodRunning) + ",status.phase!=" + string(api.PodUnknown))
	if err != nil {
		panic("terminatedSelector must compile: " + err.Error())
	}
	return selector
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
