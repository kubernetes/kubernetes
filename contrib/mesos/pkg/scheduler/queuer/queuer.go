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

package queuer

import (
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	annotation "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
)

const (
	enqueuePopTimeout  = 200 * time.Millisecond
	enqueueWaitTimeout = 1 * time.Second
	yieldPopTimeout    = 200 * time.Millisecond
	yieldWaitTimeout   = 1 * time.Second
)

type Queuer interface {
	InstallDebugHandlers(mux *http.ServeMux)
	UpdatesAvailable()
	Dequeue(id string)
	Requeue(pod *Pod)
	Reoffer(pod *Pod)

	Yield() *api.Pod

	Run(done <-chan struct{})
}

type queuer struct {
	lock            sync.Mutex       // shared by condition variables of this struct
	updates         queue.FIFO       // queue of pod updates to be processed
	queue           *queue.DelayFIFO // queue of pods to be scheduled
	deltaCond       sync.Cond        // pod changes are available for processing
	unscheduledCond sync.Cond        // there are unscheduled pods for processing
}

func New(queue *queue.DelayFIFO, updates queue.FIFO) Queuer {
	q := &queuer{
		queue:   queue,
		updates: updates,
	}
	q.deltaCond.L = &q.lock
	q.unscheduledCond.L = &q.lock
	return q
}

func (q *queuer) InstallDebugHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/debug/scheduler/podqueue", func(w http.ResponseWriter, r *http.Request) {
		for _, x := range q.queue.List() {
			if _, err := io.WriteString(w, fmt.Sprintf("%+v\n", x)); err != nil {
				break
			}
		}
	})
	mux.HandleFunc("/debug/scheduler/podstore", func(w http.ResponseWriter, r *http.Request) {
		for _, x := range q.updates.List() {
			if _, err := io.WriteString(w, fmt.Sprintf("%+v\n", x)); err != nil {
				break
			}
		}
	})
}

// signal that there are probably pod updates waiting to be processed
func (q *queuer) UpdatesAvailable() {
	q.deltaCond.Broadcast()
}

// delete a pod from the to-be-scheduled queue
func (q *queuer) Dequeue(id string) {
	q.queue.Delete(id)
}

// re-add a pod to the to-be-scheduled queue, will not overwrite existing pod data (that
// may have already changed).
func (q *queuer) Requeue(pod *Pod) {
	// use KeepExisting in case the pod has already been updated (can happen if binding fails
	// due to constraint voilations); we don't want to overwrite a newer entry with stale data.
	q.queue.Add(pod, queue.KeepExisting)
	q.unscheduledCond.Broadcast()
}

// same as Requeue but calls podQueue.Offer instead of podQueue.Add
func (q *queuer) Reoffer(pod *Pod) {
	// use KeepExisting in case the pod has already been updated (can happen if binding fails
	// due to constraint voilations); we don't want to overwrite a newer entry with stale data.
	if q.queue.Offer(pod, queue.KeepExisting) {
		q.unscheduledCond.Broadcast()
	}
}

// spawns a go-routine to watch for unscheduled pods and queue them up
// for scheduling. returns immediately.
func (q *queuer) Run(done <-chan struct{}) {
	go runtime.Until(func() {
		log.Info("Watching for newly created pods")
		q.lock.Lock()
		defer q.lock.Unlock()

		for {
			// limit blocking here for short intervals so that scheduling
			// may proceed even if there have been no recent pod changes
			p := q.updates.Await(enqueuePopTimeout)
			if p == nil {
				signalled := runtime.After(q.deltaCond.Wait)
				// we've yielded the lock
				select {
				case <-time.After(enqueueWaitTimeout):
					q.deltaCond.Broadcast() // abort Wait()
					<-signalled             // wait for lock re-acquisition
					log.V(4).Infoln("timed out waiting for a pod update")
				case <-signalled:
					// we've acquired the lock and there may be
					// changes for us to process now
				}
				continue
			}

			pod := p.(*Pod)
			if recoverAssignedSlave(pod.Pod) != "" {
				log.V(3).Infof("dequeuing assigned pod for scheduling: %v", pod.Pod.Name)
				q.Dequeue(pod.GetUID())
			} else {
				// use ReplaceExisting because we are always pushing the latest state
				now := time.Now()
				pod.deadline = &now
				if q.queue.Offer(pod, queue.ReplaceExisting) {
					q.unscheduledCond.Broadcast()
					log.V(3).Infof("queued pod for scheduling: %v", pod.Pod.Name)
				} else {
					log.Warningf("failed to queue pod for scheduling: %v", pod.Pod.Name)
				}
			}
		}
	}, 1*time.Second, done)
}

// implementation of scheduling plugin's NextPod func; see k8s plugin/pkg/scheduler
func (q *queuer) Yield() *api.Pod {
	log.V(2).Info("attempting to yield a pod")
	q.lock.Lock()
	defer q.lock.Unlock()

	for {
		// limit blocking here to short intervals so that we don't block the
		// enqueuer Run() routine for very long
		kpod := q.queue.Await(yieldPopTimeout)
		if kpod == nil {
			signalled := runtime.After(q.unscheduledCond.Wait)
			// lock is yielded at this point and we're going to wait for either
			// a timeout, or a signal that there's data
			select {
			case <-time.After(yieldWaitTimeout):
				q.unscheduledCond.Broadcast() // abort Wait()
				<-signalled                   // wait for the go-routine, and the lock
				log.V(4).Infoln("timed out waiting for a pod to yield")
			case <-signalled:
				// we have acquired the lock, and there
				// may be a pod for us to pop now
			}
			continue
		}

		pod := kpod.(*Pod).Pod
		if podName, err := cache.MetaNamespaceKeyFunc(pod); err != nil {
			log.Warningf("yield unable to understand pod object %+v, will skip: %v", pod, err)
		} else if !q.updates.Poll(podName, queue.POP_EVENT) {
			log.V(1).Infof("yield popped a transitioning pod, skipping: %+v", pod)
		} else if recoverAssignedSlave(pod) != "" {
			// should never happen if enqueuePods is filtering properly
			log.Warningf("yield popped an already-scheduled pod, skipping: %+v", pod)
		} else {
			return pod
		}
	}
}

// recoverAssignedSlave recovers the assigned Mesos slave from a pod by searching
// the BindingHostKey. For tasks in the registry of the scheduler, the same
// value is stored in T.Spec.AssignedSlave. Before launching, the BindingHostKey
// annotation is added and the executor will eventually persist that to the
// apiserver on binding.
func recoverAssignedSlave(pod *api.Pod) string {
	return pod.Annotations[annotation.BindingHostKey]
}
