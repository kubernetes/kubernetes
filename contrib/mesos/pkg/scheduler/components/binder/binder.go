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

package binder

import (
	"fmt"
	"strconv"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	annotation "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

type Binder interface {
	Bind(task *podtask.T, spec *podtask.Spec) error
}

type binder struct {
	sched scheduler.Scheduler
}

func New(sched scheduler.Scheduler) Binder {
	return &binder{
		sched: sched,
	}
}

// assumes that: caller has acquired scheduler lock and that the task is still pending
//
// Bind does not actually do the binding itself, but launches the pod as a Mesos task. The
// kubernetes executor on the slave will finally do the binding. This is different from the
// upstream scheduler in the sense that the upstream scheduler does the binding and the
// kubelet will notice that and launches the pod.
func (b *binder) Bind(task *podtask.T, spec *podtask.Spec) (err error) {
	// By this time, there is a chance that the slave is disconnected.
	if offer, ok := b.sched.Offers().Get(spec.OfferID.GetValue()); !ok || offer.HasExpired() {
		// already rescinded or timed out or otherwise invalidated
		return fmt.Errorf("failed prior to launchTask due to expired offer for task %v", task.ID)
	}

	if err = b.prepareTaskForLaunch(task, spec); err == nil {
		log.V(2).Infof(
			"launching task: %q on target %q slave %q for pod \"%v/%v\", resources %v",
			task.ID, spec.AssignedSlave, spec.SlaveID, task.Pod.Namespace, task.Pod.Name, spec.Resources,
		)

		if err = b.sched.LaunchTask(task, spec); err == nil {
			b.sched.Offers().Invalidate(spec.OfferID.GetValue())
			task.Set(podtask.Launched)
			if err = b.sched.Tasks().Update(task); err != nil {
				// this should only happen if the task has been removed or has changed status,
				// which SHOULD NOT HAPPEN as long as we're synchronizing correctly
				log.Errorf("failed to update task w/ Launched status: %v", err)
			}
			return
		}
	}
	return fmt.Errorf("Failed to launch task %v: %v", task.ID, err)
}

//TODO(jdef) unit test this, ensure that task's copy of api.Pod is not modified
func (b *binder) prepareTaskForLaunch(task *podtask.T, spec *podtask.Spec) error {
	pod := task.Pod

	// we make an effort here to avoid making changes to the task's copy of the pod, since
	// we want that to reflect the initial user spec, and not the modified spec that we
	// build for the executor to consume.
	oemCt := pod.Spec.Containers
	pod.Spec.Containers = append([]api.Container{}, oemCt...) // (shallow) clone before mod

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	task.SaveRecoveryInfo(pod.Annotations, spec)
	pod.Annotations[annotation.BindingHostKey] = spec.AssignedSlave

	for _, entry := range spec.PortMap {
		oemPorts := pod.Spec.Containers[entry.ContainerIdx].Ports
		ports := append([]api.ContainerPort{}, oemPorts...)
		p := &ports[entry.PortIdx]
		p.HostPort = int(entry.OfferPort)
		op := strconv.FormatUint(entry.OfferPort, 10)
		pod.Annotations[fmt.Sprintf(annotation.PortMappingKeyFormat, p.Protocol, p.ContainerPort)] = op
		if p.Name != "" {
			pod.Annotations[fmt.Sprintf(annotation.PortNameMappingKeyFormat, p.Protocol, p.Name)] = op
		}
		pod.Spec.Containers[entry.ContainerIdx].Ports = ports
	}

	// the kubelet-executor uses this to instantiate the pod
	log.V(3).Infof("prepared pod spec: %+v", pod)

	data, err := api.Codec.Encode(&pod)
	if err != nil {
		log.V(2).Infof("Failed to marshal the pod spec: %v", err)
		return err
	}
	spec.Data = data
	return nil
}
