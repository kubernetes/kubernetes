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
)

// NewDefaultProcurement returns the default procurement strategy that combines validation
// and responsible Mesos resource procurement. c and m are resource quantities written into
// k8s api.Pod.Spec's that don't declare resources (all containers in k8s-mesos require cpu
// and memory limits).
func NewDefaultProcurement(c mresource.CPUShares, m mresource.MegaBytes) Procurement {
	resourceProcurer := &RequirePodResources{
		defaultContainerCPULimit: c,
		defaultContainerMemLimit: m,
	}
	return AllOrNothingProcurement([]Procurement{
		ValidateProcurement,
		NodeProcurement,
		resourceProcurer.Procure,
		PortsProcurement,
	}).Procure
}

// Procurement funcs allocate resources for a task from an offer.
// Both the task and/or offer may be modified.
type Procurement func(*T, *mesos.Offer) error

// AllOrNothingProcurement provides a convenient wrapper around multiple Procurement
// objectives: the failure of any Procurement in the set results in Procure failing.
// see AllOrNothingProcurement.Procure
type AllOrNothingProcurement []Procurement

// Procure runs each Procurement in the receiver list. The first Procurement func that
// fails triggers T.Reset() and the error is returned, otherwise returns nil.
func (a AllOrNothingProcurement) Procure(t *T, offer *mesos.Offer) error {
	for _, p := range a {
		if err := p(t, offer); err != nil {
			t.Reset()
			return err
		}
	}
	return nil
}

// ValidateProcurement checks that the offered resources are kosher, and if not panics.
// If things check out ok, t.Spec is cleared and nil is returned.
func ValidateProcurement(t *T, offer *mesos.Offer) error {
	if offer == nil {
		//programming error
		panic("offer details are nil")
	}
	t.Spec = Spec{}
	return nil
}

// NodeProcurement updates t.Spec in preparation for the task to be launched on the
// slave associated with the offer.
func NodeProcurement(t *T, offer *mesos.Offer) error {
	t.Spec.SlaveID = offer.GetSlaveId().GetValue()
	t.Spec.AssignedSlave = offer.GetHostname()
	return nil
}

type RequirePodResources struct {
	defaultContainerCPULimit mresource.CPUShares
	defaultContainerMemLimit mresource.MegaBytes
}

func (r *RequirePodResources) Procure(t *T, offer *mesos.Offer) error {
	// write resource limits into the pod spec which is transferred to the executor. From here
	// on we can expect that the pod spec of a task has proper limits for CPU and memory.
	// TODO(sttts): For a later separation of the kubelet and the executor also patch the pod on the apiserver
	// TODO(sttts): fall back to requested resources if resource limit cannot be fulfilled by the offer
	// TODO(jdef): changing the state of t.Pod here feels dirty, especially since we don't use a kosher
	// method to clone the api.Pod state in T.Clone(). This needs some love.
	_, cpuLimit, _, err := mresource.LimitPodCPU(&t.Pod, r.defaultContainerCPULimit)
	if err != nil {
		return err
	}

	_, memLimit, _, err := mresource.LimitPodMem(&t.Pod, r.defaultContainerMemLimit)
	if err != nil {
		return err
	}

	log.V(3).Infof("Recording offer(s) %s/%s against pod %v: cpu: %.2f, mem: %.2f MB", offer.Id, t.Pod.Namespace, t.Pod.Name, cpuLimit, memLimit)

	t.Spec.CPU = cpuLimit
	t.Spec.Memory = memLimit

	return nil
}

// PortsProcurement convert host port mappings into mesos port resource allocations.
func PortsProcurement(t *T, offer *mesos.Offer) error {
	// fill in port mapping
	if mapping, err := t.mapper.Generate(t, offer); err != nil {
		return err
	} else {
		ports := []uint64{}
		for _, entry := range mapping {
			ports = append(ports, entry.OfferPort)
		}
		t.Spec.PortMap = mapping
		t.Spec.Ports = ports
	}
	return nil
}
