/*
Copyright 2014 Google Inc. All rights reserved.

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

package controller

import (
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
type ReplicationManager struct {
	kubeClient client.Interface
	podControl PodControlInterface
	syncTime   <-chan time.Time

	// To allow injection of syncReplicationController for testing.
	syncHandler func(controllerSpec api.ReplicationController) error
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// createReplica creates new replicated pods according to the spec.
	createReplica(namespace string, controllerSpec api.ReplicationController)
	// deletePod deletes the pod identified by podID.
	deletePod(namespace string, podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) createReplica(namespace string, controllerSpec api.ReplicationController) {
	desiredLabels := make(labels.Set)
	for k, v := range controllerSpec.DesiredState.PodTemplate.Labels {
		desiredLabels[k] = v
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: desiredLabels,
		},
		DesiredState: controllerSpec.DesiredState.PodTemplate.DesiredState,
	}
	if _, err := r.kubeClient.Pods(namespace).Create(pod); err != nil {
		glog.Errorf("Unable to create pod replica: %v", err)
	}
}

func (r RealPodControl) deletePod(namespace, podID string) error {
	return r.kubeClient.Pods(namespace).Delete(podID)
}

// NewReplicationManager creates a new ReplicationManager.
func NewReplicationManager(kubeClient client.Interface) *ReplicationManager {
	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
		},
	}
	rm.syncHandler = rm.syncReplicationController
	return rm
}

// Run begins watching and syncing.
func (rm *ReplicationManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	resourceVersion := ""
	go util.Forever(func() { rm.watchControllers(&resourceVersion) }, period)
}

// resourceVersion is a pointer to the resource version to use/update.
func (rm *ReplicationManager) watchControllers(resourceVersion *string) {
	watching, err := rm.kubeClient.ReplicationControllers(api.NamespaceAll).Watch(
		labels.Everything(),
		labels.Everything(),
		*resourceVersion,
	)
	if err != nil {
		glog.Errorf("Unexpected failure to watch: %v", err)
		time.Sleep(5 * time.Second)
		return
	}

	for {
		select {
		case <-rm.syncTime:
			rm.synchronize()
		case event, open := <-watching.ResultChan():
			if !open {
				// watchChannel has been closed, or something else went
				// wrong with our etcd watch call. Let the util.Forever()
				// that called us call us again.
				return
			}
			glog.V(4).Infof("Got watch: %#v", event)
			rc, ok := event.Object.(*api.ReplicationController)
			if !ok {
				glog.Errorf("unexpected object: %#v", event.Object)
				continue
			}
			// If we get disconnected, start where we left off.
			*resourceVersion = rc.ResourceVersion
			// Sync even if this is a deletion event, to ensure that we leave
			// it in the desired state.
			glog.V(4).Infof("About to sync from watch: %v", rc.Name)
			if err := rm.syncHandler(*rc); err != nil {
				glog.Errorf("unexpected sync. error: %v", err)
			}
		}
	}
}

func (rm *ReplicationManager) filterActivePods(pods []api.Pod) []api.Pod {
	var result []api.Pod
	for _, value := range pods {
		if api.PodSucceeded != value.CurrentState.Status &&
			api.PodFailed != value.CurrentState.Status {
			result = append(result, value)
		}
	}
	return result
}

func (rm *ReplicationManager) syncReplicationController(controllerSpec api.ReplicationController) error {
	s := labels.Set(controllerSpec.DesiredState.ReplicaSelector).AsSelector()
	podList, err := rm.kubeClient.Pods(controllerSpec.Namespace).List(s)
	if err != nil {
		return err
	}
	filteredList := rm.filterActivePods(podList.Items)
	diff := len(filteredList) - controllerSpec.DesiredState.Replicas
	if diff < 0 {
		diff *= -1
		wait := sync.WaitGroup{}
		wait.Add(diff)
		glog.V(2).Infof("Too few replicas, creating %d\n", diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wait.Done()
				rm.podControl.createReplica(controllerSpec.Namespace, controllerSpec)
			}()
		}
		wait.Wait()
	} else if diff > 0 {
		glog.V(2).Infof("Too many replicas, deleting %d\n", diff)
		wait := sync.WaitGroup{}
		wait.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wait.Done()
				rm.podControl.deletePod(controllerSpec.Namespace, filteredList[ix].Name)
			}(i)
		}
		wait.Wait()
	}
	return nil
}

func (rm *ReplicationManager) synchronize() {
	// TODO: remove this method completely and rely on the watch.
	// Add resource version tracking to watch to make this work.
	var controllerSpecs []api.ReplicationController
	list, err := rm.kubeClient.ReplicationControllers(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	controllerSpecs = list.Items
	wg := sync.WaitGroup{}
	wg.Add(len(controllerSpecs))
	for ix := range controllerSpecs {
		go func(ix int) {
			defer wg.Done()
			glog.V(4).Infof("periodic sync of %v", controllerSpecs[ix].Name)
			err := rm.syncHandler(controllerSpecs[ix])
			if err != nil {
				glog.Errorf("Error synchronizing: %#v", err)
			}
		}(ix)
	}
	wg.Wait()
}
