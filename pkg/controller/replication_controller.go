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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored in etcd
// with actual running pods.
// TODO: Allow choice of switching between etcd/apiserver watching, or remove etcd references
// from this file completely.
type ReplicationManager struct {
	etcdHelper tools.EtcdHelper
	kubeClient client.Interface
	podControl PodControlInterface
	syncTime   <-chan time.Time

	// To allow injection of syncReplicationController for testing.
	syncHandler func(controllerSpec api.ReplicationController) error

	// To allow injection of watch creation.
	watchMaker func() (watch.Interface, error)
}

// PodControlInterface is an interface that knows how to add or delete pods
// created as an interface to allow testing.
type PodControlInterface interface {
	// createReplica creates new replicated pods according to the spec.
	createReplica(controllerSpec api.ReplicationController)
	// deletePod deletes the pod identified by podID.
	deletePod(podID string) error
}

// RealPodControl is the default implementation of PodControllerInterface.
type RealPodControl struct {
	kubeClient client.Interface
}

func (r RealPodControl) createReplica(controllerSpec api.ReplicationController) {
	labels := controllerSpec.DesiredState.PodTemplate.Labels
	// TODO: don't fail to set this label just because the map isn't created.
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

// MakeReplicationManager creates a new ReplicationManager.
func MakeReplicationManager(etcdClient tools.EtcdClient, kubeClient client.Interface) *ReplicationManager {
	rm := &ReplicationManager{
		kubeClient: kubeClient,
		etcdHelper: tools.EtcdHelper{etcdClient, api.Encoding, api.Versioning},
		podControl: RealPodControl{
			kubeClient: kubeClient,
		},
	}
	rm.syncHandler = rm.syncReplicationController
	rm.watchMaker = rm.makeAPIWatch
	return rm
}

// Run begins watching and syncing.
func (rm *ReplicationManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.watchControllers() }, period)
}

// makeEtcdWatch starts watching via etcd.
func (rm *ReplicationManager) makeEtcdWatch() (watch.Interface, error) {
	return rm.etcdHelper.WatchList("/registry/controllers", tools.Everything)
}

// makeAPIWatch starts watching via the apiserver.
func (rm *ReplicationManager) makeAPIWatch() (watch.Interface, error) {
	// TODO: Fix this ugly type assertion.
	return rm.kubeClient.(*client.Client).
		Get().
		Path("watch").
		Path("replicationControllers").
		Watch()
}

func (rm *ReplicationManager) watchControllers() {
	watching, err := rm.watchMaker()
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
			glog.Infof("Got watch: %#v", event)
			if rc, ok := event.Object.(*api.ReplicationController); !ok {
				glog.Errorf("unexpected object: %#v", event.Object)
			} else {
				rm.syncHandler(*rc)
			}
		}
	}
}

func (rm *ReplicationManager) filterActivePods(pods []api.Pod) []api.Pod {
	var result []api.Pod
	for _, value := range pods {
		if api.PodStopped != value.CurrentState.Status {
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
	if diff < 0 {
		diff *= -1
		wait := sync.WaitGroup{}
		wait.Add(diff)
		glog.Infof("Too few replicas, creating %d\n", diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wait.Done()
				rm.podControl.createReplica(controllerSpec)
			}()
		}
		wait.Wait()
	} else if diff > 0 {
		glog.Infof("Too many replicas, deleting %d\n", diff)
		wait := sync.WaitGroup{}
		wait.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wait.Done()
				rm.podControl.deletePod(filteredList[ix].ID)
			}(i)
		}
		wait.Wait()
	}
	return nil
}

func (rm *ReplicationManager) synchronize() {
	var controllerSpecs []api.ReplicationController
	err := rm.etcdHelper.ExtractList("/registry/controllers", &controllerSpecs)
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
		return
	}
	wg := sync.WaitGroup{}
	wg.Add(len(controllerSpecs))
	for ix := range controllerSpecs {
		go func(ix int) {
			defer wg.Done()
			err := rm.syncHandler(controllerSpecs[ix])
			if err != nil {
				glog.Errorf("Error synchronizing: %#v", err)
			}
		}(ix)
	}
	wg.Wait()
}
