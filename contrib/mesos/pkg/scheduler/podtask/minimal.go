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

package podtask

import (
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
)

// bogus numbers that we use to make sure that there's some set of minimal offered resources on the slave
const (
	minimalCpus = 0.01
	minimalMem  = 0.25
)

var (
	DefaultMinimalPredicate = RequireAllPredicate([]FitPredicate{
		ValidationPredicate,
		NodeSelectorPredicate,
		MinimalPodResourcesPredicate,
		PortsPredicate,
	}).Fit

	DefaultMinimalProcurement = AllOrNothingProcurement([]Procurement{
		ValidateProcurement,
		NodeProcurement,
		MinimalPodResourcesProcurement,
		PortsProcurement,
	}).Procure
)

func MinimalPodResourcesPredicate(t *T, offer *mesos.Offer) bool {
	var (
		offeredCpus float64
		offeredMem  float64
	)
	for _, resource := range offer.Resources {
		if resource.GetName() == "cpus" {
			offeredCpus = resource.GetScalar().GetValue()
		}

		if resource.GetName() == "mem" {
			offeredMem = resource.GetScalar().GetValue()
		}
	}
	log.V(4).Infof("trying to match offer with pod %v/%v: cpus: %.2f mem: %.2f MB", t.Pod.Namespace, t.Pod.Name, minimalCpus, minimalMem)
	if (minimalCpus > offeredCpus) || (minimalMem > offeredMem) {
		log.V(3).Infof("not enough resources for pod %v/%v: cpus: %.2f mem: %.2f MB", t.Pod.Namespace, t.Pod.Name, minimalCpus, minimalMem)
		return false
	}
	return true
}

func MinimalPodResourcesProcurement(t *T, details *mesos.Offer) error {
	log.V(3).Infof("Recording offer(s) %s/%s against pod %v: cpu: %.2f, mem: %.2f MB", details.Id, t.Pod.Namespace, t.Pod.Name, minimalCpus, minimalMem)
	t.Spec.CPU = minimalCpus
	t.Spec.Memory = minimalMem
	return nil
}
