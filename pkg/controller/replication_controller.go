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
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored in etcd
// with actual running pods.
// TODO: Remove the etcd dependency and re-factor in terms of a generic watch interface
type ReplicationManager struct {
	etcdClient util.EtcdClient
	kubeClient client.ClientInterface
	podControl PodControlInterface
	syncTime   <-chan time.Time

	// To allow injection of syncReplicationController for testing.
	syncHandler func(controllerSpec api.ReplicationController) error
}

// An interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	createReplica(controllerSpec api.ReplicationController)
	deletePod(podID string) error
}

type RealPodControl struct {
	kubeClient client.ClientInterface
}

func (r RealPodControl) createReplica(controllerSpec api.ReplicationController) {
	labels := controllerSpec.DesiredState.PodTemplate.Labels
	if labels != nil {
		labels["replicationController"] = controllerSpec.ID
	}
	pod := api.Pod{
		JSONBase: api.JSONBase{
			ID: fmt.Sprintf("%x", rand.Int()),
		},
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

func MakeReplicationManager(etcdClient util.EtcdClient, kubeClient client.ClientInterface) *ReplicationManager {
	rm := &ReplicationManager{
		kubeClient: kubeClient,
		etcdClient: etcdClient,
		podControl: RealPodControl{
			kubeClient: kubeClient,
		},
	}
	rm.syncHandler = func(controllerSpec api.ReplicationController) error {
		return rm.syncReplicationController(controllerSpec)
	}
	return rm
}

// Begin watching and syncing.
func (rm *ReplicationManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.watchControllers() }, period)
}

func (rm *ReplicationManager) watchControllers() {
	watchChannel := make(chan *etcd.Response)
	stop := make(chan bool)
	defer func() {
		// Ensure that the call to watch ends.
		close(stop)
	}()
	go func() {
		defer util.HandleCrash()
		defer func() {
			close(watchChannel)
		}()
		_, err := rm.etcdClient.Watch("/registry/controllers", 0, true, watchChannel, stop)
		if err != etcd.ErrWatchStoppedByUser {
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

func (rm *ReplicationManager) handleWatchResponse(response *etcd.Response) (*api.ReplicationController, error) {
	if response.Action == "set" {
		if response.Node != nil {
			var controllerSpec api.ReplicationController
			err := json.Unmarshal([]byte(response.Node.Value), &controllerSpec)
			if err != nil {
				return nil, err
			}
			return &controllerSpec, nil
		} else {
			return nil, fmt.Errorf("response node is null %#v", response)
		}
	} else if response.Action == "delete" {
		// Ensure that the final state of a replication controller is applied before it is deleted.
		// Otherwise, a replication controller could be modified and then deleted (for example, from 3 to 0
		// replicas), and it would be non-deterministic which of its pods continued to exist.
		if response.PrevNode != nil {
			var controllerSpec api.ReplicationController
			if err := json.Unmarshal([]byte(response.PrevNode.Value), &controllerSpec); err != nil {
				return nil, err
			}
			return &controllerSpec, nil
		} else {
			return nil, fmt.Errorf("previous node is null %#v", response)
		}
	}

	return nil, nil
}

func (rm *ReplicationManager) filterActivePods(pods []api.Pod) []api.Pod {
	var result []api.Pod
	for _, value := range pods {
		if strings.Index(value.CurrentState.Status, "Exit") == -1 {
			result = append(result, value)
		}
	}
	return result
}

func (rm *ReplicationManager) syncReplicationController(controllerSpec api.ReplicationController) error {
	s := labels.Set(controllerSpec.DesiredState.ReplicaSelector).AsSelector()
	podList, err := rm.kubeClient.ListPods(s)
	if err != nil {
		return err
	}
	filteredList := rm.filterActivePods(podList.Items)
	diff := len(filteredList) - controllerSpec.DesiredState.Replicas
	glog.Infof("%#v", filteredList)
	if diff < 0 {
		diff *= -1
		glog.Infof("Too few replicas, creating %d\n", diff)
		for i := 0; i < diff; i++ {
			rm.podControl.createReplica(controllerSpec)
		}
	} else if diff > 0 {
		glog.Info("Too many replicas, deleting")
		for i := 0; i < diff; i++ {
			rm.podControl.deletePod(filteredList[i].ID)
		}
	}
	return nil
}

func (rm *ReplicationManager) synchronize() {
	var controllerSpecs []api.ReplicationController
	helper := util.EtcdHelper{rm.etcdClient}
	err := helper.ExtractList("/registry/controllers", &controllerSpecs)
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	for _, controllerSpec := range controllerSpecs {
		err = rm.syncHandler(controllerSpec)
		if err != nil {
			glog.Errorf("Error synchronizing: %#v", err)
		}
	}
}
