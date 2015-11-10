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
	mresource "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

func NewDefaultPredicate(c mresource.CPUShares, m mresource.MegaBytes) FitPredicate {
	return RequireAllPredicate([]FitPredicate{
		ValidationPredicate,
		NodeSelectorPredicate,
		NewPodFitsResourcesPredicate(c, m),
		PortsPredicate,
	}).Fit
}

// FitPredicate implementations determine if the given task "fits" into offered Mesos resources.
// Neither the task or offer should be modified. Note that the node can be nil.
type FitPredicate func(*T, *mesos.Offer, *api.Node) bool

type RequireAllPredicate []FitPredicate

func (f RequireAllPredicate) Fit(t *T, offer *mesos.Offer, n *api.Node) bool {
	for _, p := range f {
		if !p(t, offer, n) {
			return false
		}
	}
	return true
}

func ValidationPredicate(t *T, offer *mesos.Offer, _ *api.Node) bool {
	return t != nil && offer != nil
}

func NodeSelectorPredicate(t *T, offer *mesos.Offer, n *api.Node) bool {
	// if the user has specified a target host, make sure this offer is for that host
	if t.Pod.Spec.NodeName != "" && offer.GetHostname() != t.Pod.Spec.NodeName {
		return false
	}

	// check the NodeSelector
	if len(t.Pod.Spec.NodeSelector) > 0 {
		if n.Labels == nil {
			return false
		}
		selector := labels.SelectorFromSet(t.Pod.Spec.NodeSelector)
		if !selector.Matches(labels.Set(n.Labels)) {
			return false
		}
	}
	return true
}

func PortsPredicate(t *T, offer *mesos.Offer, _ *api.Node) bool {
	// check ports
	if _, err := t.mapper.Generate(t, offer); err != nil {
		log.V(3).Info(err)
		return false
	}
	return true
}

func NewPodFitsResourcesPredicate(c mresource.CPUShares, m mresource.MegaBytes) func(t *T, offer *mesos.Offer, _ *api.Node) bool {
	return func(t *T, offer *mesos.Offer, _ *api.Node) bool {
		// find offered cpu and mem
		var (
			offeredCpus mresource.CPUShares
			offeredMem  mresource.MegaBytes
		)
		for _, resource := range offer.Resources {
			if resource.GetName() == "cpus" {
				offeredCpus = mresource.CPUShares(*resource.GetScalar().Value)
			}

			if resource.GetName() == "mem" {
				offeredMem = mresource.MegaBytes(*resource.GetScalar().Value)
			}
		}

		// calculate cpu and mem sum over all containers of the pod
		// TODO (@sttts): also support pod.spec.resources.limit.request
		// TODO (@sttts): take into account the executor resources
		_, cpu, _, err := mresource.CPUForPod(&t.Pod, c)
		if err != nil {
			return false
		}
		_, mem, _, err := mresource.MemForPod(&t.Pod, m)
		if err != nil {
			return false
		}

		log.V(4).Infof("trying to match offer with pod %v/%v: cpus: %.2f mem: %.2f MB", t.Pod.Namespace, t.Pod.Name, cpu, mem)
		if (cpu > offeredCpus) || (mem > offeredMem) {
			log.V(3).Infof("not enough resources for pod %v/%v: cpus: %.2f mem: %.2f MB", t.Pod.Namespace, t.Pod.Name, cpu, mem)
			return false
		}
		return true
	}
}
