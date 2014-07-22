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

package buildjob

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

type BuildJobManager struct {
	etcdClient tools.EtcdClient
	kubeClient client.Interface
	podControl PodControlInterface
	syncTime   <-chan time.Time

	syncHandler func(jobSpec api.Job) error
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// createReplica creates new replicated pods according to the spec.
	runJob(jobSpec api.Job)
	// deletePod deletes the pod identified by podID.
	deletePod(podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) runJob(controllerSpec api.ReplicationController) {
	labels := controllerSpec.DesiredState.PodTemplate.Labels
	if labels != nil {
		labels["replicationController"] = controllerSpec.ID
	}
	pod := api.Pod{
		DesiredState: controllerSpec.DesiredState.PodTemplate.DesiredState,
		Labels:       controllerSpec.DesiredState.PodTemplate.Labels,
	}
	_, err := r.kubeClient.CreatePod(pod)
	if err != nil {
		glog.Errorf("%#v\n", err)
	}
}

func (r RealPodControl) deletePod(podID string) error {
	return r.kubeClient.DeletePod(podID)
}

func MakeBuildJobManager(etcdClient tools.EtcdClient, kubeClient client.Interface) *BuildJobManager {
	rm := &BuildJobManager{
		kubeClient: kubeClient,
		etcdClient: etcdClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
		},
	}
	// TODO
	rm.syncHandler = func(jobSpec api.Job) error {
		return rm.syncJobState(jobSpec)
	}

	return rm
}

// Run begins watching and syncing.
func (rm *BuildJobManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.watchBuildJobs() }, period)
}

// watch loop
func (rm *BuildJobManager) watchBuildJobs() {
	watchChannel := make(chan *etcd.Response)
	stop := make(chan bool)
	// Ensure that the call to watch ends.
	defer close(stop)

	go func() {
		defer util.HandleCrash()
		_, err := rm.etcdClient.Watch("/jobs/build", 0, true, watchChannel, stop)
		if err == etcd.ErrWatchStoppedByUser {
			close(watchChannel)
		} else {
			glog.Errorf("etcd.Watch stopped unexpectedly: %v (%#v)", err, err)
		}
	}()

	for {
		select {
		case <-rm.syncTime:
			rm.synchronize()
		case watchResponse, open := <-watchChannel:
			if !open || watchResponse == nil {
				// watchChannel has been closed, or something else went
				// wrong with our etcd watch call. Let the util.Forever()
				// that called us call us again.
				return
			}
			glog.Infof("Got watch: %#v", watchResponse)
			controller, err := rm.handleWatchResponse(watchResponse)
			if err != nil {
				glog.Errorf("Error handling data: %#v, %#v", err, watchResponse)
				continue
			}
			rm.syncHandler(*controller)
		}
	}
}

func (rm *BuildJobManager) handleWatchResponse(response *etcd.Response) (*api.Job, error) {
	switch response.Action {
	case "set":
		if response.Node == nil {
			return nil, fmt.Errorf("response node is null %#v", response)
		}
		var jobSpec api.Job
		if err := json.Unmarshal([]byte(response.Node.Value), &jobSpec); err != nil {
			return nil, err
		}
		return &jobSpec, nil
	case "delete":
		// Ensure that the final state of a replication controller is applied before it is deleted.
		// Otherwise, a replication controller could be modified and then deleted (for example, from 3 to 0
		// replicas), and it would be non-deterministic which of its pods continued to exist.
		if response.PrevNode == nil {
			return nil, fmt.Errorf("previous node is null %#v", response)
		}
		var jobSpec api.Job
		if err := json.Unmarshal([]byte(response.PrevNode.Value), &jobSpec); err != nil {
			return nil, err
		}
		return &jobSpec, nil
	}
	return nil, nil
}

// Sync loop implementation, pointed to by rm.syncHandler
func (rm *BuildJobManager) syncJobState(jobSpec api.Job) error {
	s := labels.Set(controllerSpec.DesiredState.ReplicaSelector).AsSelector()
	podList, err := rm.kubeClient.ListPods(s)
	if err != nil {
		return err
	}
	filteredList := rm.filterActivePods(podList.Items)
	diff := len(filteredList) - controllerSpec.DesiredState.Replicas
	if diff < 0 {
		diff *= -1
		glog.Infof("Too few replicas, creating %d\n", diff)
		for i := 0; i < diff; i++ {
			rm.podControl.createReplica(controllerSpec)
		}
	} else if diff > 0 {
		glog.Infof("Too many replicas, deleting %d\n", diff)
		for i := 0; i < diff; i++ {
			rm.podControl.deletePod(filteredList[i].ID)
		}
	}
	return nil
}

func (rm *BuildJobManager) synchronize() {
	var jobSpecs []api.Job
	helper := tools.EtcdHelper{rm.etcdClient}
	err := helper.ExtractList("/jobs/build", &jobSpecs)
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, jobSpec := range jobSpecs {
		err = rm.syncHandler(jobSpec)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
