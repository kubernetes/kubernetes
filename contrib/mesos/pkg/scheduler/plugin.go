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
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	merrors "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/operations"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	types "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/types"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	plugin "k8s.io/kubernetes/plugin/pkg/scheduler"
)

const (
	pluginRecoveryDelay = 100 * time.Millisecond // delay after scheduler plugin crashes, before we resume scheduling

	FailedScheduling = "FailedScheduling"
	Scheduled        = "Scheduled"
)

type PluginInterface interface {
	// the apiserver may have a different state for the pod than we do
	// so reconcile our records, but only for this one pod
	reconcileTask(*podtask.T)

	// execute the Scheduling plugin, should start a go routine and return immediately
	Run(<-chan struct{})
}

type PluginConfig struct {
	*plugin.Config
	fw       types.Framework
	client   *client.Client
	qr       *queuer.Queuer
	deleter  *operations.Deleter
	starting chan struct{} // startup latch
}

func NewPlugin(c *PluginConfig) PluginInterface {
	return &schedulerPlugin{
		config:   c.Config,
		fw:       c.fw,
		client:   c.client,
		qr:       c.qr,
		deleter:  c.deleter,
		starting: c.starting,
	}
}

type schedulerPlugin struct {
	config   *plugin.Config
	fw       types.Framework
	client   *client.Client
	qr       *queuer.Queuer
	deleter  *operations.Deleter
	starting chan struct{}
}

func (s *schedulerPlugin) Run(done <-chan struct{}) {
	defer close(s.starting)
	go runtime.Until(s.scheduleOne, pluginRecoveryDelay, done)
}

// hacked from GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/scheduler.go,
// with the Modeler stuff removed since we don't use it because we have mesos.
func (s *schedulerPlugin) scheduleOne() {
	pod := s.config.NextPod()

	// pods which are pre-scheduled (i.e. NodeName is set) are deleted by the kubelet
	// in upstream. Not so in Mesos because the kubelet hasn't see that pod yet. Hence,
	// the scheduler has to take care of this:
	if pod.Spec.NodeName != "" && pod.DeletionTimestamp != nil {
		log.V(3).Infof("deleting pre-scheduled, not yet running pod: %s/%s", pod.Namespace, pod.Name)
		s.client.Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0))
		return
	}

	log.V(3).Infof("Attempting to schedule: %+v", pod)
	dest, err := s.config.Algorithm.Schedule(pod, s.config.NodeLister) // call kubeScheduler.Schedule
	if err != nil {
		log.V(1).Infof("Failed to schedule: %+v", pod)
		s.config.Recorder.Eventf(pod, FailedScheduling, "Error scheduling: %v", err)
		s.config.Error(pod, err)
		return
	}
	b := &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
		Target: api.ObjectReference{
			Kind: "Node",
			Name: dest,
		},
	}
	if err := s.config.Binder.Bind(b); err != nil {
		log.V(1).Infof("Failed to bind pod: %+v", err)
		s.config.Recorder.Eventf(pod, FailedScheduling, "Binding rejected: %v", err)
		s.config.Error(pod, err)
		return
	}
	s.config.Recorder.Eventf(pod, Scheduled, "Successfully assigned %v to %v", pod.Name, dest)
}

// this pod may be out of sync with respect to the API server registry:
//      this pod   |  apiserver registry
//    -------------|----------------------
//      host=.*    |  404           ; pod was deleted
//      host=.*    |  5xx           ; failed to sync, try again later?
//      host=""    |  host=""       ; perhaps no updates to process?
//      host=""    |  host="..."    ; pod has been scheduled and assigned, is there a task assigned? (check TaskIdKey in binding?)
//      host="..." |  host=""       ; pod is no longer scheduled, does it need to be re-queued?
//      host="..." |  host="..."    ; perhaps no updates to process?
//
// TODO(jdef) this needs an integration test
func (s *schedulerPlugin) reconcileTask(t *podtask.T) {
	log.V(1).Infof("reconcile pod %v, assigned to slave %q", t.Pod.Name, t.Spec.AssignedSlave)
	ctx := api.WithNamespace(api.NewDefaultContext(), t.Pod.Namespace)
	pod, err := s.client.Pods(api.NamespaceValue(ctx)).Get(t.Pod.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// attempt to delete
			if err = s.deleter.DeleteOne(&queuer.Pod{Pod: &t.Pod}); err != nil && err != merrors.NoSuchPodErr && err != merrors.NoSuchTaskErr {
				log.Errorf("failed to delete pod: %v: %v", t.Pod.Name, err)
			}
		} else {
			//TODO(jdef) other errors should probably trigger a retry (w/ backoff).
			//For now, drop the pod on the floor
			log.Warning("aborting reconciliation for pod %v: %v", t.Pod.Name, err)
		}
		return
	}

	log.Infof("pod %v scheduled on %q according to apiserver", pod.Name, pod.Spec.NodeName)
	if t.Spec.AssignedSlave != pod.Spec.NodeName {
		if pod.Spec.NodeName == "" {
			// pod is unscheduled.
			// it's possible that we dropped the pod in the scheduler error handler
			// because of task misalignment with the pod (task.Has(podtask.Launched) == true)

			podKey, err := podtask.MakePodKey(ctx, pod.Name)
			if err != nil {
				log.Error(err)
				return
			}

			s.fw.Lock()
			defer s.fw.Unlock()

			if _, state := s.fw.Tasks().ForPod(podKey); state != podtask.StateUnknown {
				//TODO(jdef) reconcile the task
				log.Errorf("task already registered for pod %v", pod.Name)
				return
			}

			now := time.Now()
			log.V(3).Infof("reoffering pod %v", podKey)
			s.qr.Reoffer(queuer.NewPodWithDeadline(pod, &now))
		} else {
			// pod is scheduled.
			// not sure how this happened behind our backs. attempt to reconstruct
			// at least a partial podtask.T record.
			//TODO(jdef) reconcile the task
			log.Errorf("pod already scheduled: %v", pod.Name)
		}
	} else {
		//TODO(jdef) for now, ignore the fact that the rest of the spec may be different
		//and assume that our knowledge of the pod aligns with that of the apiserver
		log.Error("pod reconciliation does not support updates; not yet implemented")
	}
}
