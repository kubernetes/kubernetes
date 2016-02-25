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

package framework

import (
	"fmt"
	"sync"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	unversioned_core "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
)

const podGCCoalescePeriod = 30 * time.Second

type (
	cancelFunc func()

	jobKey struct {
		host    string
		slaveID string
	}

	podGCRequest struct {
		jobKey
		// response is optional, if provided GC will send true if the GC request runs to completion, otherwise false
		response chan<- bool
	}

	podGC struct {
		inbox   chan podGCRequest
		ongoing map[jobKey]cancelFunc
		pods    unversioned_core.PodsGetter
		done    <-chan struct{}
	}
)

func newPodGC(pods unversioned_core.PodsGetter, done <-chan struct{}) *podGC {
	return &podGC{
		inbox:   make(chan podGCRequest),
		ongoing: map[jobKey]cancelFunc{},
		pods:    pods,
		done:    done,
	}
}

// schedule returns a response chan that generates true if GC completed normaly, otherwise false.
// a response chan may generate multiple values but only the first one is valid.
func (gc *podGC) schedule(host, slaveID string) (chan<- bool, error) {
	var (
		err      error
		response = make(chan bool, 2)
	)
	if host != "" && slaveID != "" {
		select {
		case gc.inbox <- podGCRequest{jobKey{host, slaveID}, response}:
			return response, nil
		case <-gc.done:
		}
	} else {
		err = fmt.Errorf("failed to schedule GC for host %q slaveID %q: both params are required", host, slaveID)
	}
	response <- false
	close(response)
	return response, err
}

func (gc *podGC) run() {
	var (
		t        = time.NewTimer(0)
		requests = map[jobKey][]podGCRequest{}
		timeChan <-chan time.Time
	)
	defer func() {
		t.Stop()
		// close of all abort signals for ongoing cleanup
		for k, cancel := range gc.ongoing {
			cancel()
			delete(gc.ongoing, k)
		}
	}()

	<-t.C // sanity: start w/ a clear timer chan
	for {
		select {
		case <-gc.done:
			return

		case r := <-gc.inbox:
			// are we coalescing?
			if timeChan == nil {
				log.V(1).Infof("will GC pods for host %q slave %q momentarily", r.host, r.slaveID)
				t.Reset(podGCCoalescePeriod)
				timeChan = t.C
			}
			key := jobKey{host: r.host, slaveID: r.slaveID}
			existing := requests[key]
			requests[key] = append(existing, r)

		case <-timeChan:
			// if we got this far then there are GC jobs on deck
			timeChan = nil

			var (
				abortOnce sync.Once
				abortChan = make(chan struct{})
				abortFunc = func() { abortOnce.Do(func() { close(abortChan) }) }
			)
			for key, v := range requests {
				cancel := gc.ongoing[key]
				if cancel != nil {
					// cancel already-running jobs for the same {host,slave}
					cancel()
				}
				gc.ongoing[key] = cancelFunc(func() {
					defer abortFunc()
					for i := range v {
						v[i].response <- false
					}
				})
			}

			go gc.do(requests, abortChan)

			// reset coalesence
			requests = map[jobKey][]podGCRequest{}
		}
	}
}

func (gc *podGC) do(requests map[jobKey][]podGCRequest, abort <-chan struct{}) {
	type podKey struct{ namespace, name string }

mainLoop:
	for k, req := range requests {
		// short-circuit
		select {
		case <-abort:
			respondTo(req, false)
			continue
		default:
		}

		log.V(1).Infof("garbage collecting pods on host %q slave %q", k.host, k.slaveID)

		// get a snapshot of all the pods on the node
		sel := fields.OneTermEqualSelector(client.PodHost, k.host)
		list, err := gc.pods.Pods(api.NamespaceAll).List(api.ListOptions{FieldSelector: sel})
		if err != nil {
			respondTo(req, false)
			continue
		}

		// filter them further, according to slaveID:
		// - delete the ones that match
		// - ignore not-found errors
		for i := range list.Items {
			if slaveID, ok := list.Items[i].Annotations[meta.SlaveIdKey]; ok && slaveID == k.slaveID {
				var (
					ns   = list.Items[i].Namespace
					name = list.Items[i].Name
				)
				log.V(1).Infof("deleting pod %s/%s", ns, name)
				if err := gc.pods.Pods(ns).Delete(name, api.NewDeleteOptions(0)); err != nil && !errors.IsNotFound(err) {
					log.Errorf("failed to delete pod %v/%v, aborting GC: %v", ns, name, err)
					respondTo(req, false)
					continue mainLoop
				}
			}
		}

		// success!
		respondTo(req, true)
	}
}

func respondTo(requests []podGCRequest, val bool) {
	for i := range requests {
		requests[i].response <- val
	}
}
