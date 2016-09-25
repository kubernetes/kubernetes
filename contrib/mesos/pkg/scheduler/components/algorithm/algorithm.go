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

package algorithm

import (
	"fmt"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/algorithm/podschedulers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
)

// SchedulerAlgorithm is the interface that orchestrates the pod scheduling.
//
// Schedule implements the Scheduler interface of Kubernetes.
// It returns the selectedMachine's hostname or an error if the schedule failed.
type SchedulerAlgorithm interface {
	Schedule(pod *api.Pod) (string, error)
}

// SchedulerAlgorithm implements the algorithm.ScheduleAlgorithm interface
type schedulerAlgorithm struct {
	sched        scheduler.Scheduler
	podUpdates   queue.FIFO
	podScheduler podschedulers.PodScheduler
	taskConfig   podtask.Config
	defaultCpus  resources.CPUShares
	defaultMem   resources.MegaBytes
}

// New returns a new SchedulerAlgorithm
// TODO(sur): refactor params to separate config object
func New(
	sched scheduler.Scheduler,
	podUpdates queue.FIFO,
	podScheduler podschedulers.PodScheduler,
	taskConfig podtask.Config,
	defaultCpus resources.CPUShares,
	defaultMem resources.MegaBytes,
) SchedulerAlgorithm {
	return &schedulerAlgorithm{
		sched:        sched,
		podUpdates:   podUpdates,
		podScheduler: podScheduler,
		taskConfig:   taskConfig,
		defaultCpus:  defaultCpus,
		defaultMem:   defaultMem,
	}
}

func (k *schedulerAlgorithm) Schedule(pod *api.Pod) (string, error) {
	log.Infof("Try to schedule pod %v\n", pod.Name)
	ctx := api.WithNamespace(api.NewDefaultContext(), pod.Namespace)

	// default upstream scheduler passes pod.Name as binding.PodID
	podKey, err := podtask.MakePodKey(ctx, pod.Name)
	if err != nil {
		return "", err
	}

	k.sched.Lock()
	defer k.sched.Unlock()

	switch task, state := k.sched.Tasks().ForPod(podKey); state {
	case podtask.StateUnknown:
		// There's a bit of a potential race here, a pod could have been yielded() and
		// then before we get *here* it could be deleted.
		// We use meta to index the pod in the store since that's what k8s reflector does.
		podName, err := cache.MetaNamespaceKeyFunc(pod)
		if err != nil {
			log.Warningf("aborting Schedule, unable to understand pod object %+v", pod)
			return "", errors.NoSuchPodErr
		}

		if deleted := k.podUpdates.Poll(podName, queue.DELETE_EVENT); deleted {
			// avoid scheduling a pod that's been deleted between yieldPod() and Schedule()
			log.Infof("aborting Schedule, pod has been deleted %+v", pod)
			return "", errors.NoSuchPodErr
		}

		// write resource limits into the pod spec.
		// From here on we can expect that the pod spec of a task has proper limits for CPU and memory.
		k.limitPod(pod)

		podTask, err := podtask.New(ctx, k.taskConfig, pod)
		if err != nil {
			log.Warningf("aborting Schedule, unable to create podtask object %+v: %v", pod, err)
			return "", err
		}

		podTask, err = k.sched.Tasks().Register(podTask)
		if err != nil {
			return "", err
		}

		return k.doSchedule(podTask)

	//TODO(jdef) it's possible that the pod state has diverged from what
	//we knew previously, we should probably update the task.Pod state here
	//before proceeding with scheduling
	case podtask.StatePending:
		if pod.UID != task.Pod.UID {
			// we're dealing with a brand new pod spec here, so the old one must have been
			// deleted -- and so our task store is out of sync w/ respect to reality
			//TODO(jdef) reconcile task
			return "", fmt.Errorf("task %v spec is out of sync with pod %v spec, aborting schedule", task.ID, pod.Name)
		} else if task.Has(podtask.Launched) {
			// task has been marked as "launched" but the pod binding creation may have failed in k8s,
			// but we're going to let someone else handle it, probably the mesos task error handler
			return "", fmt.Errorf("task %s has already been launched, aborting schedule", task.ID)
		} else {
			return k.doSchedule(task)
		}

	default:
		return "", fmt.Errorf("task %s is not pending, nothing to schedule", task.ID)
	}
}

// limitPod limits the given pod based on the scheduler's default limits.
func (k *schedulerAlgorithm) limitPod(pod *api.Pod) error {
	cpuRequest, cpuLimit, _, err := resources.LimitPodCPU(pod, k.defaultCpus)
	if err != nil {
		return err
	}

	memRequest, memLimit, _, err := resources.LimitPodMem(pod, k.defaultMem)
	if err != nil {
		return err
	}

	log.V(3).Infof(
		"setting pod %s/%s resources: requested cpu %.2f mem %.2f MB, limited cpu %.2f mem %.2f MB",
		pod.Namespace, pod.Name, cpuRequest, memRequest, cpuLimit, memLimit,
	)

	return nil
}

// doSchedule implements the actual scheduling of the given pod task.
// It checks whether the offer has been accepted and is still present in the offer registry.
// It delegates to the actual pod scheduler and updates the task registry.
func (k *schedulerAlgorithm) doSchedule(task *podtask.T) (string, error) {
	var offer offers.Perishable
	var err error

	if task.HasAcceptedOffer() {
		// verify that the offer is still on the table
		var ok bool
		offer, ok = k.sched.Offers().Get(task.GetOfferId())

		if !ok || offer.HasExpired() {
			task.Offer.Release()
			task.Reset()
			if err = k.sched.Tasks().Update(task); err != nil {
				return "", err
			}
		}
	}

	var spec *podtask.Spec
	if offer == nil {
		offer, spec, err = k.podScheduler.SchedulePod(k.sched.Offers(), task)
	}

	if err != nil {
		return "", err
	}

	details := offer.Details()
	if details == nil {
		return "", fmt.Errorf("offer already invalid/expired for task %v", task.ID)
	}

	if task.Offer != nil && task.Offer != offer {
		return "", fmt.Errorf("task.offer assignment must be idempotent, task %+v: offer %+v", task, offer)
	}

	task.Offer = offer
	task.Spec = spec

	if err := k.sched.Tasks().Update(task); err != nil {
		offer.Release()
		return "", err
	}

	return details.GetHostname(), nil
}
