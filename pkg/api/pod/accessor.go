/*
Copyright The Kubernetes Authors.

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

package pod

import (
	fmt "fmt"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/kubernetes/pkg/apis/core"
)

// PodAccessor provides read access to core.Pod fields without
// requiring the caller to assert a concrete *core.Pod struct.
//
// Values returned by PodAccessor getters (such as slices, pointers, or maps)
// must not be mutated directly, and callers must not assume that direct
// mutations will be reflected in the underlying object.
//
// +k8s:deepcopy-gen=false
type PodAccessor interface {
	runtime.Object
	v1.Object
	GetSpecInitContainers() []core.Container
	GetSpecContainers() []core.Container
	GetSpecEphemeralContainers() []core.EphemeralContainer
	GetSpecRestartPolicy() core.RestartPolicy
	GetSpecTerminationGracePeriodSeconds() *int64
	GetSpecServiceAccountName() string
	GetSpecNodeName() string
	GetSpecHostNetwork() bool
	GetSpecAffinity() *core.Affinity
	GetSpecSchedulerName() string
	GetStatusPhase() core.PodPhase
	GetStatusNominatedNodeName() string
	GetStatusHostIPs() []core.HostIP
	GetStatusPodIPs() []core.PodIP
	GetStatusResourceClaimStatuses() []core.PodResourceClaimStatus
	GetStatusExtendedResourceClaimStatus() *core.PodExtendedResourceClaimStatus
	GetStatusNodeAllocatableResourceClaimStatuses() []core.NodeAllocatableResourceClaimStatus
}

type podAccessor struct {
	*core.Pod
}

var _ PodAccessor = (*podAccessor)(nil)

// Accessor returns a PodAccessor for the provided object, or an error if the object
// does not implement PodAccessor or *core.Pod.
func Accessor(obj any) (PodAccessor, error) {
	switch t := obj.(type) {
	case PodAccessor:
		return t, nil
	case *core.Pod:
		if t == nil {
			return nil, nil
		}
		return &podAccessor{Pod: t}, nil
	default:
		return nil, fmt.Errorf("object %T does not implement PodAccessor", obj)
	}
}

// MustAccessor returns a PodAccessor for the provided object, or panics if the object
// does not implement PodAccessor or *core.Pod.
func MustAccessor(obj any) PodAccessor {
	acc, err := Accessor(obj)
	if err != nil {
		panic(err)
	}
	return acc
}

func (in *podAccessor) GetSpecInitContainers() []core.Container {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.InitContainers
}

func (in *podAccessor) GetSpecContainers() []core.Container {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.Containers
}

func (in *podAccessor) GetSpecEphemeralContainers() []core.EphemeralContainer {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.EphemeralContainers
}

func (in *podAccessor) GetSpecRestartPolicy() core.RestartPolicy {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.RestartPolicy
}

func (in *podAccessor) GetSpecTerminationGracePeriodSeconds() *int64 {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.TerminationGracePeriodSeconds
}

func (in *podAccessor) GetSpecServiceAccountName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.ServiceAccountName
}

func (in *podAccessor) GetSpecNodeName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.NodeName
}

func (in *podAccessor) GetSpecHostNetwork() bool {
	if in == nil || in.Pod == nil {
		return false
	}
	return in.Pod.Spec.HostNetwork
}

func (in *podAccessor) GetSpecAffinity() *core.Affinity {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Spec.Affinity
}

func (in *podAccessor) GetSpecSchedulerName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Spec.SchedulerName
}

func (in *podAccessor) GetStatusPhase() core.PodPhase {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Status.Phase
}

func (in *podAccessor) GetStatusNominatedNodeName() string {
	if in == nil || in.Pod == nil {
		return ""
	}
	return in.Pod.Status.NominatedNodeName
}

func (in *podAccessor) GetStatusHostIPs() []core.HostIP {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.HostIPs
}

func (in *podAccessor) GetStatusPodIPs() []core.PodIP {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.PodIPs
}

func (in *podAccessor) GetStatusResourceClaimStatuses() []core.PodResourceClaimStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.ResourceClaimStatuses
}

func (in *podAccessor) GetStatusExtendedResourceClaimStatus() *core.PodExtendedResourceClaimStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.ExtendedResourceClaimStatus
}

func (in *podAccessor) GetStatusNodeAllocatableResourceClaimStatuses() []core.NodeAllocatableResourceClaimStatus {
	if in == nil || in.Pod == nil {
		return nil
	}
	return in.Pod.Status.NodeAllocatableResourceClaimStatuses
}
