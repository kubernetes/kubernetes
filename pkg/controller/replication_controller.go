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
	"fmt"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
type ReplicationManager struct {
	kubeClient client.Interface
	podControl PodControlInterface
	syncTime   <-chan time.Time

	// To allow injection of syncReplicationController for testing.
	syncHandler func(controller api.ReplicationController) error
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// createReplica creates new replicated pods according to the spec.
	createReplica(namespace string, controller api.ReplicationController)
	// deletePod deletes the pod identified by podID.
	deletePod(namespace string, podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	kubeClient client.Interface
	recorder   record.EventRecorder
}

// Time period of main replication controller sync loop
const DefaultSyncPeriod = 5 * time.Second

func (r RealPodControl) createReplica(namespace string, controller api.ReplicationController) {
	desiredLabels := make(labels.Set)
	for k, v := range controller.Spec.Template.Labels {
		desiredLabels[k] = v
	}
	desiredAnnotations := make(labels.Set)
	for k, v := range controller.Spec.Template.Annotations {
		desiredAnnotations[k] = v
	}

	// use the dash (if the name isn't too long) to make the pod name a bit prettier
	prefix := fmt.Sprintf("%s-", controller.Name)
	if ok, _ := validation.ValidatePodName(prefix, true); !ok {
		prefix = controller.Name
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       desiredLabels,
			Annotations:  desiredAnnotations,
			GenerateName: prefix,
		},
	}
	if err := api.Scheme.Convert(&controller.Spec.Template.Spec, &pod.Spec); err != nil {
		util.HandleError(fmt.Errorf("unable to convert pod template: %v", err))
		return
	}
	if labels.Set(pod.Labels).AsSelector().Empty() {
		util.HandleError(fmt.Errorf("unable to create pod replica, no labels"))
		return
	}
	if _, err := r.kubeClient.Pods(namespace).Create(pod); err != nil {
		r.recorder.Eventf(&controller, "failedCreate", "Error creating: %v", err)
		util.HandleError(fmt.Errorf("unable to create pod replica: %v", err))
	}
}

func (r RealPodControl) deletePod(namespace, podID string) error {
	return r.kubeClient.Pods(namespace).Delete(podID)
}

// NewReplicationManager creates a new ReplicationManager.
func NewReplicationManager(kubeClient client.Interface) *ReplicationManager {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
			recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "replication-controller"}),
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
		fields.Everything(),
		*resourceVersion,
	)
	if err != nil {
		util.HandleError(fmt.Errorf("unable to watch: %v", err))
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
				// wrong with our watch call. Let the util.Forever()
				// that called us call us again.
				return
			}
			if event.Type == watch.Error {
				util.HandleError(fmt.Errorf("error from watch during sync: %v", errors.FromObject(event.Object)))
				// Clear the resource version, this may cause us to skip some elements on the watch,
				// but we'll catch them on the synchronize() call, so it works out.
				*resourceVersion = ""
				continue
			}
			glog.V(4).Infof("Got watch: %#v", event)
			rc, ok := event.Object.(*api.ReplicationController)
			if !ok {
				if status, ok := event.Object.(*api.Status); ok {
					if status.Status == api.StatusFailure {
						glog.Errorf("failed to watch: %v", status)
						// Clear resource version here, as above, this won't hurt consistency, but we
						// should consider introspecting more carefully here. (or make the apiserver smarter)
						// "why not both?"
						*resourceVersion = ""
						continue
					}
				}
				util.HandleError(fmt.Errorf("unexpected object: %#v", event.Object))
				continue
			}
			// If we get disconnected, start where we left off.
			*resourceVersion = rc.ResourceVersion
			// Sync even if this is a deletion event, to ensure that we leave
			// it in the desired state.
			glog.V(4).Infof("About to sync from watch: %v", rc.Name)
			if err := rm.syncHandler(*rc); err != nil {
				util.HandleError(fmt.Errorf("unexpected sync error: %v", err))
			}
		}
	}
}

// Helper function. Also used in pkg/registry/controller, for now.
func FilterActivePods(pods []api.Pod) []*api.Pod {
	var result []*api.Pod
	for i := range pods {
		value := &pods[i]
		if api.PodSucceeded != value.Status.Phase &&
			api.PodFailed != value.Status.Phase {
			result = append(result, value)
		}
	}
	return result
}

func (rm *ReplicationManager) syncReplicationController(controller api.ReplicationController) error {
	s := labels.Set(controller.Spec.Selector).AsSelector()
	podList, err := rm.kubeClient.Pods(controller.Namespace).List(s)
	if err != nil {
		return err
	}
	filteredList := FilterActivePods(podList.Items)
	activePods := len(filteredList)
	diff := activePods - controller.Spec.Replicas
	if diff < 0 {
		diff *= -1
		wait := sync.WaitGroup{}
		wait.Add(diff)
		glog.V(2).Infof("Too few \"%s\" replicas, creating %d\n", controller.Name, diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wait.Done()
				rm.podControl.createReplica(controller.Namespace, controller)
			}()
		}
		wait.Wait()
	} else if diff > 0 {
		glog.V(2).Infof("Too many \"%s\" replicas, deleting %d\n", controller.Name, diff)
		wait := sync.WaitGroup{}
		wait.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wait.Done()
				rm.podControl.deletePod(controller.Namespace, filteredList[ix].Name)
			}(i)
		}
		wait.Wait()
	}
	if controller.Status.Replicas != activePods {
		controller.Status.Replicas = activePods
		_, err = rm.kubeClient.ReplicationControllers(controller.Namespace).Update(&controller)
		if err != nil {
			return err
		}
	}
	return nil
}

func (rm *ReplicationManager) synchronize() {
	// TODO: remove this method completely and rely on the watch.
	// Add resource version tracking to watch to make this work.
	var controllers []api.ReplicationController
	list, err := rm.kubeClient.ReplicationControllers(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		util.HandleError(fmt.Errorf("synchronization error: %v", err))
		return
	}
	controllers = list.Items
	wg := sync.WaitGroup{}
	wg.Add(len(controllers))
	for ix := range controllers {
		go func(ix int) {
			defer wg.Done()
			glog.V(4).Infof("periodic sync of %v", controllers[ix].Name)
			err := rm.syncHandler(controllers[ix])
			if err != nil {
				util.HandleError(fmt.Errorf("error synchronizing: %v", err))
			}
		}(ix)
	}
	wg.Wait()
}
