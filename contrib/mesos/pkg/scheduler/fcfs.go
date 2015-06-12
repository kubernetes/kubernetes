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
	"fmt"
	log "github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/offers"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/scheduler/podtask"
)

// A first-come-first-serve scheduler: acquires the first offer that can support the task
func FCFSScheduleFunc(r offers.Registry, unused SlaveIndex, task *podtask.T) (offers.Perishable, error) {
	podName := fmt.Sprintf("%s/%s", task.Pod.Namespace, task.Pod.Name)
	var acceptedOffer offers.Perishable
	err := r.Walk(func(p offers.Perishable) (bool, error) {
		offer := p.Details()
		if offer == nil {
			return false, fmt.Errorf("nil offer while scheduling task %v", task.ID)
		}
		if task.AcceptOffer(offer) {
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
	return nil, noSuitableOffersErr
}
