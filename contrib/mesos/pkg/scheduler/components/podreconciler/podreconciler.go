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

package podreconciler

import (
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/deleter"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/queuer"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
)

// PodReconciler reconciles a pod with the apiserver
type PodReconciler interface {
	Reconcile(t *podtask.T)
}

type podReconciler struct {
	sched   scheduler.Scheduler
	client  *clientset.Clientset
	qr      queuer.Queuer
	deleter deleter.Deleter
}

func New(sched scheduler.Scheduler, client *clientset.Clientset, qr queuer.Queuer, deleter deleter.Deleter) PodReconciler {
	return &podReconciler{
		sched:   sched,
		client:  client,
		qr:      qr,
		deleter: deleter,
	}
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
func (s *podReconciler) Reconcile(t *podtask.T) {
	log.V(1).Infof("reconcile pod %v, assigned to slave %q", t.Pod.Name, t.Spec.AssignedSlave)
	ctx := api.WithNamespace(api.NewDefaultContext(), t.Pod.Namespace)
	pod, err := s.client.Core().Pods(api.NamespaceValue(ctx)).Get(t.Pod.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// attempt to delete
			if err = s.deleter.DeleteOne(&queuer.Pod{Pod: &t.Pod}); err != nil && err != errors.NoSuchPodErr && err != errors.NoSuchTaskErr {
				log.Errorf("failed to delete pod: %v: %v", t.Pod.Name, err)
			}
		} else {
			//TODO(jdef) other errors should probably trigger a retry (w/ backoff).
			//For now, drop the pod on the floor
			log.Warningf("aborting reconciliation for pod %v: %v", t.Pod.Name, err)
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

			s.sched.Lock()
			defer s.sched.Unlock()

			if _, state := s.sched.Tasks().ForPod(podKey); state != podtask.StateUnknown {
				//TODO(jdef) reconcile the task
				log.Errorf("task already registered for pod %v", pod.Name)
				return
			}

			now := time.Now()
			log.V(3).Infof("reoffering pod %v", podKey)
			s.qr.Reoffer(queuer.NewPod(pod, queuer.Deadline(now)))
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
