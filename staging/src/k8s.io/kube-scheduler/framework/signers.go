/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// This file contains the names and implementations of various functions used
// to compute pod signatures for use in batching and other scheduling optimizations.
// See the definition of BatchablePlugin for more details.

// Signer names
const (
	DynamicResourcesSignerName = "v1.Pod.Spec.DynamicResources"
	ImageNamesSignerName       = "v1.Pod.Spec.CanonicalImageNames()"
	LabelsSignerName           = "v1.Pod.Labels"
	NodeNameSignerName         = "v1.Pod.Spec.NodeName"
	NodeAffinitySignerName     = "v1.Pod.Spec.Affinity.NodeAffinity"
	NodeSelectorSignerName     = "v1.Pod.Spec.Affinity.NodeSelector"
	HostPortsSignerName        = "v1.Pod.Spec.HostPorts()"
	ResourcesSignerName        = "v1.Pod.Spec.ContainerRequestsAndOverheads()"
	SchedulerNameSignerName    = "v1.Pod.Spec.SchedulerName"
	TolerationsSignerName      = "v1.Pod.Spec.Tolerations"
	VolumesSignerName          = "v1.Pod.Spec.Volumes.NonSyntheticSources()"
	FeaturesSignerName         = "v1.Pod.Spec.RequiredFeatures()"
)

// Common signers. These are either generic or shared across plugins.

func HostPortsSigner(pod *v1.Pod) any {
	portSet := sets.New[int32]()
	containers := []v1.Container{}
	containers = append(containers, pod.Spec.Containers...)
	containers = append(containers, pod.Spec.InitContainers...)
	for _, container := range containers {
		for _, port := range container.Ports {
			if port.HostPort != 0 {
				portSet.Insert(port.HostPort)
			}
		}
	}
	ports := portSet.UnsortedList()
	sort.Slice(ports, func(i, j int) bool {
		return ports[i] < ports[j]
	})
	return ports
}

func NodeSelectorRequirementSigner(reqs v1.NodeSelectorRequirement) v1.NodeSelectorRequirement {
	newValues := make([]string, len(reqs.Values))
	copy(newValues, reqs.Values)
	sort.Slice(newValues, func(i, j int) bool {
		return newValues[i] < newValues[j]
	})
	return v1.NodeSelectorRequirement{
		Key:      reqs.Key,
		Operator: reqs.Operator,
		Values:   newValues,
	}
}

func NodeSelectorRequirementsSigner(reqs []v1.NodeSelectorRequirement) []v1.NodeSelectorRequirement {
	newReqs := make([]v1.NodeSelectorRequirement, len(reqs))
	copy(newReqs, reqs)
	sort.Slice(newReqs, func(i, j int) bool {
		return newReqs[i].Key < newReqs[j].Key || (newReqs[i].Key == newReqs[j].Key && newReqs[i].Operator < newReqs[j].Operator)
	})
	return newReqs
}

func NodeSelectorTermSigner(t *v1.NodeSelectorTerm) v1.NodeSelectorTerm {
	return v1.NodeSelectorTerm{
		MatchExpressions: NodeSelectorRequirementsSigner(t.MatchExpressions),
		MatchFields:      NodeSelectorRequirementsSigner(t.MatchFields),
	}
}

func PreferredSchedulingTermSigner(terms []v1.PreferredSchedulingTerm) ([]string, error) {
	newTerms := make([]string, len(terms))
	for i, t := range terms {
		termStr, err := json.Marshal(v1.PreferredSchedulingTerm{
			Weight:     t.Weight,
			Preference: NodeSelectorTermSigner(&t.Preference),
		})
		if err != nil {
			return nil, err
		}
		newTerms[i] = string(termStr)
	}
	sort.Slice(newTerms, func(i, j int) bool {
		return newTerms[i] < newTerms[j]
	})
	return newTerms, nil
}

func NodeSelectorTermsSigner(terms []v1.NodeSelectorTerm) ([]string, error) {
	req := make([]string, len(terms))
	for i, t := range terms {
		tStr, err := json.Marshal(NodeSelectorTermSigner(&t))
		if err != nil {
			return nil, err
		}
		req[i] = string(tStr)
	}

	sort.Slice(req, func(i, j int) bool {
		return req[i] < req[j]
	})

	return req, nil
}

type nodeAffinitySignerResult struct {
	Required  []string
	Preferred []string
}

func NodeAffinitySigner(pod *v1.Pod) (any, error) {
	if pod.Spec.Affinity != nil {
		if pod.Spec.Affinity.NodeAffinity != nil {
			n := pod.Spec.Affinity.NodeAffinity
			pref, err := PreferredSchedulingTermSigner(n.PreferredDuringSchedulingIgnoredDuringExecution)
			if err != nil {
				return nil, err
			}

			req, err := NodeSelectorTermsSigner(n.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms)
			if err != nil {
				return nil, err
			}

			return nodeAffinitySignerResult{
				Required:  req,
				Preferred: pref,
			}, nil
		}
	}
	return nil, nil
}

func TolerationsSigner(pod *v1.Pod) any {
	ret := []v1.Toleration{}
	ret = append(ret, pod.Spec.Tolerations...)
	sort.Slice(ret, func(i, j int) bool {
		return ret[i].Key < ret[j].Key || (ret[i].Key == ret[j].Key && ret[i].Value < ret[j].Value)
	})
	return ret
}

// We special case volumes because config and secret volumes don't
// impact scheduling but are very specific to individual pods. If we
// don't exclude them no pods will have matching signatures.
func VolumesSigner(pod *v1.Pod) any {
	ret := []string{}
	for _, vol := range pod.Spec.Volumes {
		if vol.VolumeSource.ConfigMap == nil && vol.VolumeSource.Secret == nil {
			volStr, err := json.Marshal(vol.VolumeSource)
			if err != nil {
				return nil
			}
			ret = append(ret, string(volStr))
		}
	}
	sort.Slice(ret, func(i, j int) bool {
		return ret[i] < ret[j]
	})
	return ret
}
