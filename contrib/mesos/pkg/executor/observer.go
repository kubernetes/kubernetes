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

package executor

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
)

// taskUpdateTx execute a task update transaction f for the task identified by
// taskId. if no such task exists then f is not invoked and an error is
// returned. if f is invoked then taskUpdateTx returns the bool result of f.
type taskUpdateTx func(taskId string, f func(*kuberTask, *api.Pod) bool) (changed bool, err error)

// podObserver receives callbacks for every pod state change on the apiserver and
// for each decides whether to execute a task update transaction.
type podObserver struct {
	podController *framework.Controller
	terminate     <-chan struct{}
	taskUpdateTx  taskUpdateTx
}

func newPodObserver(podLW cache.ListerWatcher, taskUpdateTx taskUpdateTx, terminate <-chan struct{}) *podObserver {
	// watch pods from the given pod ListWatch
	if podLW == nil {
		// fail early to make debugging easier
		panic("cannot create executor with nil PodLW")
	}

	p := &podObserver{
		terminate:    terminate,
		taskUpdateTx: taskUpdateTx,
	}
	_, p.podController = framework.NewInformer(podLW, &api.Pod{}, podRelistPeriod, &framework.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod := obj.(*api.Pod)
			log.V(4).Infof("pod %s/%s created on apiserver", pod.Namespace, pod.Name)
			p.handleChangedApiserverPod(pod)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			pod := newObj.(*api.Pod)
			log.V(4).Infof("pod %s/%s updated on apiserver", pod.Namespace, pod.Name)
			p.handleChangedApiserverPod(pod)
		},
		DeleteFunc: func(obj interface{}) {
			pod := obj.(*api.Pod)
			log.V(4).Infof("pod %s/%s deleted on apiserver", pod.Namespace, pod.Name)
		},
	})
	return p
}

// run begins observing pod state changes; blocks until the terminate chan closes.
func (p *podObserver) run() {
	p.podController.Run(p.terminate)
}

// handleChangedApiserverPod is invoked for pod add/update state changes and decides whether
// task updates are necessary. if so, a task update is executed via taskUpdateTx.
func (p *podObserver) handleChangedApiserverPod(pod *api.Pod) {
	// Don't do anything for pods without task anotation which means:
	// - "pre-scheduled" pods which have a NodeName set to this node without being scheduled already.
	// - static/mirror pods: they'll never have a TaskID annotation, and we don't expect them to ever change.
	// - all other pods that haven't passed through the launch-task-binding phase, which would set annotations.
	taskId := pod.Annotations[meta.TaskIdKey]
	if taskId == "" {
		// There also could be a race between the overall launch-task process and this update, but here we
		// will never be able to process such a stale update because the "update pod" that we're receiving
		// in this func won't yet have a task ID annotation. It follows that we can safely drop such a stale
		// update on the floor because we'll get another update later that, in addition to the changes that
		// we're dropping now, will also include the changes from the binding process.
		log.V(5).Infof("ignoring pod update for %s/%s because %s annotation is missing", pod.Namespace, pod.Name, meta.TaskIdKey)
		return
	}

	_, err := p.taskUpdateTx(taskId, func(_ *kuberTask, relatedPod *api.Pod) (sendSnapshot bool) {
		if relatedPod == nil {
			// should never happen because:
			// (a) the update pod record has already passed through the binding phase in launchTasks()
			// (b) all remaining updates to executor.{pods,tasks} are sync'd in unison
			log.Errorf("internal state error: pod not found for task %s", taskId)
			return
		}

		// TODO(sttts): check whether we can and should send all "semantic" changes down to the kubelet
		// see kubelet/config/config.go for semantic change detection

		// check for updated labels/annotations: need to forward these for the downward API
		sendSnapshot = sendSnapshot || updateMetaMap(&relatedPod.Labels, pod.Labels)
		sendSnapshot = sendSnapshot || updateMetaMap(&relatedPod.Annotations, pod.Annotations)

		// terminating pod?
		if pod.Status.Phase == api.PodRunning {
			timeModified := differentTime(relatedPod.DeletionTimestamp, pod.DeletionTimestamp)
			graceModified := differentPeriod(relatedPod.DeletionGracePeriodSeconds, pod.DeletionGracePeriodSeconds)
			if timeModified || graceModified {
				log.Infof("pod %s/%s is terminating at %v with %vs grace period, telling kubelet",
					pod.Namespace, pod.Name, *pod.DeletionTimestamp, *pod.DeletionGracePeriodSeconds)

				// modify the pod in our registry instead of sending the new pod. The later
				// would allow that other changes bleed into the kubelet. For now we are
				// very conservative changing this behaviour.
				relatedPod.DeletionTimestamp = pod.DeletionTimestamp
				relatedPod.DeletionGracePeriodSeconds = pod.DeletionGracePeriodSeconds
				sendSnapshot = true
			}
		}
		return
	})
	if err != nil {
		log.Errorf("failed to update pod %s/%s: %+v", pod.Namespace, pod.Name, err)
	}
}

// updateMetaMap looks for differences between src and dest; if there are differences
// then dest is changed (possibly to point to src) and this func returns true.
func updateMetaMap(dest *map[string]string, src map[string]string) (changed bool) {
	// check for things in dest that are missing in src
	for k := range *dest {
		if _, ok := src[k]; !ok {
			changed = true
			break
		}
	}
	if !changed {
		if len(*dest) == 0 {
			if len(src) > 0 {
				changed = true
				goto finished
			}
			// no update needed
			return
		}
		// check for things in src that are missing/different in dest
		for k, v := range src {
			if vv, ok := (*dest)[k]; !ok || vv != v {
				changed = true
				break
			}
		}
	}
finished:
	*dest = src
	return
}

func differentTime(a, b *unversioned.Time) bool {
	return (a == nil) != (b == nil) || (a != nil && b != nil && *a != *b)
}

func differentPeriod(a, b *int64) bool {
	return (a == nil) != (b == nil) || (a != nil && b != nil && *a != *b)
}
