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

package binder

import (
	"fmt"
	"strconv"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	annotation "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

type Binder interface {
	Bind(binding *api.Binding) error
}

type binder struct {
	sched scheduler.Scheduler
}

func New(sched scheduler.Scheduler) Binder {
	return &binder{
		sched: sched,
	}
}

// implements binding.Registry, launches the pod-associated-task in mesos
func (b *binder) Bind(binding *api.Binding) error {

	ctx := api.WithNamespace(api.NewContext(), binding.Namespace)

	// default upstream scheduler passes pod.Name as binding.Name
	podKey, err := podtask.MakePodKey(ctx, binding.Name)
	if err != nil {
		return err
	}

	b.sched.Lock()
	defer b.sched.Unlock()

	switch task, state := b.sched.Tasks().ForPod(podKey); state {
	case podtask.StatePending:
		return b.bind(ctx, binding, task)
	default:
		// in this case it's likely that the pod has been deleted between Schedule
		// and Bind calls
		log.Infof("No pending task for pod %s", podKey)
		return errors.NoSuchPodErr //TODO(jdef) this error is somewhat misleading since the task could be running?!
	}
}

func (b *binder) rollback(task *podtask.T, err error) error {
	task.Offer.Release()
	task.Reset()
	if err2 := b.sched.Tasks().Update(task); err2 != nil {
		log.Errorf("failed to update pod task: %v", err2)
	}
	return err
}

// assumes that: caller has acquired scheduler lock and that the task is still pending
//
// bind does not actually do the binding itself, but launches the pod as a Mesos task. The
// kubernetes executor on the slave will finally do the binding. This is different from the
// upstream scheduler in the sense that the upstream scheduler does the binding and the
// kubelet will notice that and launches the pod.
func (b *binder) bind(ctx api.Context, binding *api.Binding, task *podtask.T) (err error) {
	// sanity check: ensure that the task hasAcceptedOffer(), it's possible that between
	// Schedule() and now that the offer for this task was rescinded or invalidated.
	// ((we should never see this here))
	if !task.HasAcceptedOffer() {
		return fmt.Errorf("task has not accepted a valid offer %v", task.ID)
	}

	// By this time, there is a chance that the slave is disconnected.
	offerId := task.GetOfferId()
	if offer, ok := b.sched.Offers().Get(offerId); !ok || offer.HasExpired() {
		// already rescinded or timed out or otherwise invalidated
		return b.rollback(task, fmt.Errorf("failed prior to launchTask due to expired offer for task %v", task.ID))
	}

	if err = b.prepareTaskForLaunch(ctx, binding.Target.Name, task, offerId); err == nil {
		log.V(2).Infof(
			"launching task: %q on target %q slave %q for pod \"%v/%v\", resources %v",
			task.ID, binding.Target.Name, task.Spec.SlaveID, task.Pod.Namespace, task.Pod.Name, task.Spec.Resources,
		)

		if err = b.sched.LaunchTask(task); err == nil {
			b.sched.Offers().Invalidate(offerId)
			task.Set(podtask.Launched)
			if err = b.sched.Tasks().Update(task); err != nil {
				// this should only happen if the task has been removed or has changed status,
				// which SHOULD NOT HAPPEN as long as we're synchronizing correctly
				log.Errorf("failed to update task w/ Launched status: %v", err)
			}
			return
		}
	}
	return b.rollback(task, fmt.Errorf("Failed to launch task %v: %v", task.ID, err))
}

//TODO(jdef) unit test this, ensure that task's copy of api.Pod is not modified
func (b *binder) prepareTaskForLaunch(ctx api.Context, machine string, task *podtask.T, offerId string) error {
	pod := task.Pod

	// we make an effort here to avoid making changes to the task's copy of the pod, since
	// we want that to reflect the initial user spec, and not the modified spec that we
	// build for the executor to consume.
	oemCt := pod.Spec.Containers
	pod.Spec.Containers = append([]api.Container{}, oemCt...) // (shallow) clone before mod

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	task.SaveRecoveryInfo(pod.Annotations)
	pod.Annotations[annotation.BindingHostKey] = task.Spec.AssignedSlave

	for _, entry := range task.Spec.PortMap {
		oemPorts := pod.Spec.Containers[entry.ContainerIdx].Ports
		ports := append([]api.ContainerPort{}, oemPorts...)
		p := &ports[entry.PortIdx]
		p.HostPort = int32(entry.OfferPort)
		op := strconv.FormatUint(entry.OfferPort, 10)
		pod.Annotations[fmt.Sprintf(annotation.PortMappingKeyFormat, p.Protocol, p.ContainerPort)] = op
		if p.Name != "" {
			pod.Annotations[fmt.Sprintf(annotation.PortNameMappingKeyFormat, p.Protocol, p.Name)] = op
		}
		pod.Spec.Containers[entry.ContainerIdx].Ports = ports
	}

	// the kubelet-executor uses this to instantiate the pod
	log.V(3).Infof("prepared pod spec: %+v", pod)

	data, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), &pod)
	if err != nil {
		log.V(2).Infof("Failed to marshal the pod spec: %v", err)
		return err
	}
	task.Spec.Data = data
	return nil
}
