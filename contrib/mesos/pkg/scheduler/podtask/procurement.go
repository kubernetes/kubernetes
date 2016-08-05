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

package podtask

import (
	"fmt"
	"math"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/executorinfo"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

// NewDefaultProcurement returns the default procurement strategy that combines validation
// and responsible Mesos resource procurement. c and m are resource quantities written into
// k8s api.Pod.Spec's that don't declare resources (all containers in k8s-mesos require cpu
// and memory limits).
func NewDefaultProcurement(prototype *mesos.ExecutorInfo, eir executorinfo.Registry) Procurement {
	return AllOrNothingProcurement([]Procurement{
		NewNodeProcurement(),
		NewPodResourcesProcurement(),
		NewPortsProcurement(),
		NewExecutorResourceProcurer(prototype.GetResources(), eir),
	})
}

// Procurement is the interface that implements resource procurement.
//
// Procure procurs offered resources for a given pod task T
// on a given node and stores the procurement result.
//
// Initially the procurement pipe contains an initial empty Spec
// and the the complete Mesos offer. As the procurement pipeline progresses
// the specified resources go up as they are being procured
// while the remaining Mesos offer resources go down until they are depleted.
//
// It returns an error if the procurement failed.
//
// Note that the T struct also includes a Spec field.
// This differs from the procured Spec which is meant to be filled
// by a chain of Procure invocations (procurement pipeline).
//
// In contrast T.Spec is meant not to be filled by the procurement chain
// but rather by a final scheduler instance.
//
// api.Node is an optional (possibly nil) param.
type Procurement interface {
	Procure(*T, *api.Node, *ProcureState) error
}

// ProcureState holds the current state of the procurement pipeline.
// It contains the pod launch specification and the Mesos offer
// from which resources are being procured.
type ProcureState struct {
	offer *mesos.Offer // source
	spec  *Spec        // sink
}

// Result returns the procurement result consisting
// of the procured pod specification and the remaining
// Mesos offer.
func (ps *ProcureState) Result() (*Spec, *mesos.Offer) {
	return ps.spec, ps.offer
}

// NewProcureState returns an ProcureState containing an empty Spec
// and a deep copy of the given offer.
func NewProcureState(offer *mesos.Offer) *ProcureState {
	return &ProcureState{
		spec:  &Spec{},
		offer: proto.Clone(offer).(*mesos.Offer),
	}
}

// The ProcurementFunc type is an adapter to use ordinary functions as Procurement implementations.
type ProcurementFunc func(*T, *api.Node, *ProcureState) error

func (p ProcurementFunc) Procure(t *T, n *api.Node, ps *ProcureState) error {
	return p(t, n, ps)
}

// AllOrNothingProcurement provides a convenient wrapper around multiple Procurement
// objectives: the failure of any Procurement in the set results in Procure failing.
// see AllOrNothingProcurement.Procure
type AllOrNothingProcurement []Procurement

// Procure runs each Procurement in the receiver list. The first Procurement func that
// fails triggers T.Reset() and the error is returned, otherwise returns nil.
func (a AllOrNothingProcurement) Procure(t *T, n *api.Node, ps *ProcureState) error {
	for _, p := range a {
		err := p.Procure(t, n, ps)
		if err != nil {
			return err
		}
	}
	return nil
}

// NewNodeProcurement returns a Procurement that checks whether the given pod task and offer
// have valid node informations available and whether the pod spec node selector matches
// the pod labels.
// If the check is successful the slave ID and assigned slave is set in the given Spec.
func NewNodeProcurement() Procurement {
	return ProcurementFunc(func(t *T, n *api.Node, ps *ProcureState) error {
		// if the user has specified a target host, make sure this offer is for that host
		if t.Pod.Spec.NodeName != "" && ps.offer.GetHostname() != t.Pod.Spec.NodeName {
			return fmt.Errorf(
				"NodeName %q does not match offer hostname %q",
				t.Pod.Spec.NodeName, ps.offer.GetHostname(),
			)
		}

		// check the NodeSelector
		if len(t.Pod.Spec.NodeSelector) > 0 {
			// *api.Node is optional for procurement
			if n == nil || n.Labels == nil {
				return fmt.Errorf(
					"NodeSelector %v does not match empty labels of pod %s/%s",
					t.Pod.Spec.NodeSelector, t.Pod.Namespace, t.Pod.Name,
				)
			}
			selector := labels.SelectorFromSet(t.Pod.Spec.NodeSelector)
			if !selector.Matches(labels.Set(n.Labels)) {
				return fmt.Errorf(
					"NodeSelector %v does not match labels %v of pod %s/%s",
					t.Pod.Spec.NodeSelector, t.Pod.Labels, t.Pod.Namespace, t.Pod.Name,
				)
			}
		}

		ps.spec.SlaveID = ps.offer.GetSlaveId().GetValue()
		ps.spec.AssignedSlave = ps.offer.GetHostname()

		return nil
	})
}

// NewPodResourcesProcurement converts k8s pod cpu and memory resource requirements into
// mesos resource allocations.
func NewPodResourcesProcurement() Procurement {
	return ProcurementFunc(func(t *T, _ *api.Node, ps *ProcureState) error {
		// TODO(sttts): fall back to requested resources if resource limit cannot be fulfilled by the offer
		_, limits, err := api.PodRequestsAndLimits(&t.Pod)
		if err != nil {
			return err
		}

		wantedCpus := float64(resources.NewCPUShares(limits[api.ResourceCPU]))
		wantedMem := float64(resources.NewMegaBytes(limits[api.ResourceMemory]))

		log.V(4).Infof(
			"trying to match offer with pod %v/%v: cpus: %.2f mem: %.2f MB",
			t.Pod.Namespace, t.Pod.Name, wantedCpus, wantedMem,
		)

		podRoles := t.Roles()
		procuredCpu, remaining := procureScalarResources("cpus", wantedCpus, podRoles, ps.offer.GetResources())
		if procuredCpu == nil {
			return fmt.Errorf(
				"not enough cpu resources for pod %s/%s: want=%v",
				t.Pod.Namespace, t.Pod.Name, wantedCpus,
			)
		}

		procuredMem, remaining := procureScalarResources("mem", wantedMem, podRoles, remaining)
		if procuredMem == nil {
			return fmt.Errorf(
				"not enough mem resources for pod %s/%s: want=%v",
				t.Pod.Namespace, t.Pod.Name, wantedMem,
			)
		}

		ps.offer.Resources = remaining
		ps.spec.Resources = append(ps.spec.Resources, append(procuredCpu, procuredMem...)...)
		return nil
	})
}

// NewPortsProcurement returns a Procurement procuring ports
func NewPortsProcurement() Procurement {
	return ProcurementFunc(func(t *T, _ *api.Node, ps *ProcureState) error {
		// fill in port mapping
		if mapping, err := t.mapper.Map(&t.Pod, t.Roles(), ps.offer); err != nil {
			return err
		} else {
			ports := []resources.Port{}
			for _, entry := range mapping {
				ports = append(ports, resources.Port{
					Port: entry.OfferPort,
					Role: entry.Role,
				})
			}
			ps.spec.PortMap = mapping
			ps.spec.Resources = append(ps.spec.Resources, resources.PortRanges(ports)...)
		}
		return nil
	})
}

// NewExecutorResourceProcurer returns a Procurement procuring executor resources
// If a given offer has no executor IDs set, the given prototype executor resources are considered for procurement.
// If a given offer has one executor ID set, only pod resources are being procured.
// An offer with more than one executor ID implies an invariant violation and the first executor ID is being considered.
func NewExecutorResourceProcurer(rs []*mesos.Resource, registry executorinfo.Registry) Procurement {
	return ProcurementFunc(func(t *T, _ *api.Node, ps *ProcureState) error {
		eids := len(ps.offer.GetExecutorIds())
		switch {
		case eids == 0:
			wantedCpus := resources.Sum(resources.Filter(rs, resources.IsScalar, resources.HasName("cpus")))
			wantedMem := resources.Sum(resources.Filter(rs, resources.IsScalar, resources.HasName("mem")))

			procuredCpu, remaining := procureScalarResources("cpus", wantedCpus, t.FrameworkRoles, ps.offer.GetResources())
			if procuredCpu == nil {
				return fmt.Errorf("not enough cpu resources for executor: want=%v", wantedCpus)
			}

			procuredMem, remaining := procureScalarResources("mem", wantedMem, t.FrameworkRoles, remaining)
			if procuredMem == nil {
				return fmt.Errorf("not enough mem resources for executor: want=%v", wantedMem)
			}

			ps.offer.Resources = remaining
			ps.spec.Executor = registry.New(ps.offer.GetHostname(), append(procuredCpu, procuredMem...))
			return nil

		case eids == 1:
			e, err := registry.Get(ps.offer.GetHostname())
			if err != nil {
				return err
			}
			ps.spec.Executor = e
			return nil

		default:
			// offers with more than 1 ExecutorId should be rejected by the
			// framework long before they arrive here.
			return fmt.Errorf("got offer with more than 1 executor id: %v", ps.offer.GetExecutorIds())
		}
	})
}

// smallest number such that 1.0 + epsilon != 1.0
// see https://github.com/golang/go/issues/966
var epsilon = math.Nextafter(1, 2) - 1

// procureScalarResources procures offered resources that
// 1. Match the given name
// 2. Match the given roles
// 3. The given wanted scalar value can be fully consumed by offered resources
// Roles are being considered in the specified roles slice ordering.
func procureScalarResources(
	name string,
	want float64,
	roles []string,
	offered []*mesos.Resource,
) (procured, remaining []*mesos.Resource) {
	sorted := resources.ByRoles(roles...).Sort(offered)
	procured = make([]*mesos.Resource, 0, len(sorted))
	remaining = make([]*mesos.Resource, 0, len(sorted))

	for _, r := range sorted {
		if want >= epsilon && resources.MatchesAll(r, resources.HasName(name), resources.IsScalar) {
			left, role := r.GetScalar().GetValue(), r.Role
			consumed := math.Min(want, left)

			want -= consumed
			left -= consumed

			if left >= epsilon {
				r = mesosutil.NewScalarResource(name, left)
				r.Role = role
				remaining = append(remaining, r)
			}

			consumedRes := mesosutil.NewScalarResource(name, consumed)
			consumedRes.Role = role
			procured = append(procured, consumedRes)
		} else {
			remaining = append(remaining, r)
		}
	}

	// demanded value (want) was not fully consumed violating invariant 3.
	// thus no resources must be procured
	if want >= epsilon {
		return nil, offered
	}

	return
}
