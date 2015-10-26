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

package scheduler

import (
	"sync"

	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podschedulers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

type mesosFramework struct {
	sync.Mutex
	mesosScheduler *MesosScheduler
}

func (fw *mesosFramework) PodScheduler() podschedulers.PodScheduler {
	return fw.mesosScheduler.podScheduler
}

func (fw *mesosFramework) Offers() offers.Registry {
	return fw.mesosScheduler.offers
}

func (fw *mesosFramework) Tasks() podtask.Registry {
	return fw.mesosScheduler.taskRegistry
}

func (fw *mesosFramework) CreatePodTask(ctx api.Context, pod *api.Pod) (*podtask.T, error) {
	return podtask.New(ctx, "", *pod, fw.mesosScheduler.executor)
}

func (fw *mesosFramework) SlaveHostNameFor(id string) string {
	return fw.mesosScheduler.slaveHostNames.HostName(id)
}

func (fw *mesosFramework) KillTask(taskId string) error {
	killTaskId := mutil.NewTaskID(taskId)
	_, err := fw.mesosScheduler.driver.KillTask(killTaskId)
	return err
}

func (fw *mesosFramework) LaunchTask(task *podtask.T) error {
	// assume caller is holding scheduler lock
	taskList := []*mesos.TaskInfo{task.BuildTaskInfo()}
	offerIds := []*mesos.OfferID{task.Offer.Details().Id}
	filters := &mesos.Filters{}
	_, err := fw.mesosScheduler.driver.LaunchTasks(offerIds, taskList, filters)
	return err
}