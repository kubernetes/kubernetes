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

package podschedulers

import (
	"fmt"

	log "github.com/golang/glog"

	"github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
)

type fcfsPodScheduler struct {
	procurement podtask.Procurement
	lookupNode  node.LookupFunc
}

func NewFCFSPodScheduler(pr podtask.Procurement, lookupNode node.LookupFunc) PodScheduler {
	return &fcfsPodScheduler{pr, lookupNode}
}

// A first-come-first-serve scheduler: acquires the first offer that can support the task
func (fps *fcfsPodScheduler) SchedulePod(r offers.Registry, task *podtask.T) (offers.Perishable, *podtask.Spec, error) {
	podName := fmt.Sprintf("%s/%s", task.Pod.Namespace, task.Pod.Name)
	var matchingOffer offers.Perishable
	var acceptedSpec *podtask.Spec
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

		ps := podtask.NewProcureState(offer)
		err := fps.procurement.Procure(task, n, ps)
		if err != nil {
			log.V(5).Infof(
				"Offer %q does not fit pod %s/%s: %v",
				offer.Id, task.Pod.Namespace, task.Pod.Name, err,
			)
			return false, nil // continue
		}

		if !p.Acquire() {
			log.V(2).Infof(
				"Could not acquire offer %q for pod %s/%s",
				offer.Id, task.Pod.Namespace, task.Pod.Name,
			)
			return false, nil // continue
		}

		matchingOffer = p
		acceptedSpec, _ = ps.Result()
		log.V(3).Infof("Pod %s accepted offer %v", podName, offer.Id.GetValue())
		return true, nil // stop, we found an offer
	})
	if matchingOffer != nil {
		if err != nil {
			log.Warningf("problems walking the offer registry: %v, attempting to continue", err)
		}
		return matchingOffer, acceptedSpec, nil
	}
	if err != nil {
		log.V(2).Infof("failed to find a fit for pod: %s, err = %v", podName, err)
		return nil, nil, err
	}
	log.V(2).Infof("failed to find a fit for pod: %s", podName)
	return nil, nil, errors.NoSuitableOffersErr
}

func (fps *fcfsPodScheduler) Fit(t *podtask.T, offer *mesosproto.Offer, n *api.Node) bool {
	return fps.procurement.Procure(t, n, podtask.NewProcureState(offer)) == nil
}
