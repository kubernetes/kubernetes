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

package podschedulers

import (
	"fmt"

	log "github.com/golang/glog"

	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
)

type allocationStrategy struct {
	fitPredicate podtask.FitPredicate
	procurement  podtask.Procurement
}

func (a *allocationStrategy) FitPredicate() podtask.FitPredicate {
	return a.fitPredicate
}

func (a *allocationStrategy) Procurement() podtask.Procurement {
	return a.procurement
}

func NewAllocationStrategy(fitPredicate podtask.FitPredicate, procurement podtask.Procurement) AllocationStrategy {
	if fitPredicate == nil {
		panic("fitPredicate is required")
	}
	if procurement == nil {
		panic("procurement is required")
	}
	return &allocationStrategy{
		fitPredicate: fitPredicate,
		procurement:  procurement,
	}
}

type fcfsPodScheduler struct {
	AllocationStrategy
	lookupNode node.LookupFunc
}

func NewFCFSPodScheduler(as AllocationStrategy, lookupNode node.LookupFunc) PodScheduler {
	return &fcfsPodScheduler{as, lookupNode}
}

// A first-come-first-serve scheduler: acquires the first offer that can support the task
func (fps *fcfsPodScheduler) SchedulePod(r offers.Registry, task *podtask.T) (offers.Perishable, error) {
	podName := fmt.Sprintf("%s/%s", task.Pod.Namespace, task.Pod.Name)
	var acceptedOffer offers.Perishable
	err := r.Walk(func(p offers.Perishable) (bool, error) {
		offer := p.Details()
		if offer == nil {
			return false, fmt.Errorf("nil offer while scheduling task %v", task.ID)
		}

		// check that the node actually exists. As offers are declined if not, the
		// case n==nil can only happen when the node object was deleted since the
		// offer came in.
		nodeName := offer.GetHostname()
		n := fps.lookupNode(nodeName)
		if n == nil {
			log.V(3).Infof("ignoring offer for node %s because node went away", nodeName)
			return false, nil
		}

		if fps.FitPredicate()(task, offer, n) {
			if p.Acquire() {
				acceptedOffer = p
				log.V(3).Infof("Pod %s accepted offer %v", podName, offer.Id.GetValue())
				return true, nil // stop, we found an offer
			}
		}
		return false, nil // continue
	})
	if acceptedOffer != nil {
		if err != nil {
			log.Warningf("problems walking the offer registry: %v, attempting to continue", err)
		}
		return acceptedOffer, nil
	}
	if err != nil {
		log.V(2).Infof("failed to find a fit for pod: %s, err = %v", podName, err)
		return nil, err
	}
	log.V(2).Infof("failed to find a fit for pod: %s", podName)
	return nil, errors.NoSuitableOffersErr
}
