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
)

type MesosFramework struct {
	sync.Mutex
	MesosScheduler *MesosScheduler
}

func (fw *MesosFramework) PodScheduler() podschedulers.PodScheduler {
	return fw.MesosScheduler.podScheduler
}

func (fw *MesosFramework) Offers() offers.Registry {
	return fw.MesosScheduler.offers
}

func (fw *MesosFramework) Tasks() podtask.Registry {
	return fw.MesosScheduler.taskRegistry
}

func (fw *MesosFramework) SlaveHostNameFor(id string) string {
	return fw.MesosScheduler.slaveHostNames.HostName(id)
}

func (fw *MesosFramework) KillTask(taskId string) error {
	killTaskId := mutil.NewTaskID(taskId)
	_, err := fw.MesosScheduler.driver.KillTask(killTaskId)
	return err
}

func (fw *MesosFramework) LaunchTask(task *podtask.T) error {
	// assume caller is holding scheduler lock
	ei := fw.MesosScheduler.executor
	taskList := []*mesos.TaskInfo{task.BuildTaskInfo(ei)}
	offerIds := []*mesos.OfferID{task.Offer.Details().Id}
	filters := &mesos.Filters{}
	_, err := fw.MesosScheduler.driver.LaunchTasks(offerIds, taskList, filters)
	return err
}
