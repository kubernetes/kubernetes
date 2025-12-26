/*
Copyright 2019 The Kubernetes Authors.

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

package podtopologyspread

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	v1helper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/utils/ptr"
)

// topologySpreadConstraint is an internal version for v1.TopologySpreadConstraint
// and where the selector is parsed.
// Fields are exported for comparison during testing.
type topologySpreadConstraint struct {
	MaxSkew            int32
	TopologyKey        string
	Selector           labels.Selector
	MinDomains         int32
	NodeAffinityPolicy v1.NodeInclusionPolicy
	NodeTaintsPolicy   v1.NodeInclusionPolicy
}

func (tsc *topologySpreadConstraint) matchNodeInclusionPolicies(logger klog.Logger, pod *v1.Pod, node *v1.Node, require nodeaffinity.RequiredNodeAffinity, enableComparisonOperators bool) bool {
	if tsc.NodeAffinityPolicy == v1.NodeInclusionPolicyHonor {
		// We ignore parsing errors here for backwards compatibility.
		if match, _ := require.Match(node); !match {
			return false
		}
	}

	if tsc.NodeTaintsPolicy == v1.NodeInclusionPolicyHonor {
		if _, untolerated := v1helper.FindMatchingUntoleratedTaint(logger, node.Spec.Taints, pod.Spec.Tolerations, helper.DoNotScheduleTaintsFilterFunc(), enableComparisonOperators); untolerated {
			return false
		}
	}
	return true
}

// buildDefaultConstraints builds the constraints for a pod using
// .DefaultConstraints and the selectors from the services, replication
// controllers, replica sets and stateful sets that match the pod.
func (pl *PodTopologySpread) buildDefaultConstraints(p *v1.Pod, action v1.UnsatisfiableConstraintAction) ([]topologySpreadConstraint, error) {
	constraints, err := pl.filterTopologySpreadConstraints(pl.defaultConstraints, p.Labels, action)
	if err != nil || len(constraints) == 0 {
		return nil, err
	}
	selector := helper.DefaultSelector(p, pl.services, pl.replicationCtrls, pl.replicaSets, pl.statefulSets)
	if selector.Empty() {
		return nil, nil
	}
	for i := range constraints {
		constraints[i].Selector = selector
	}
	return constraints, nil
}

// nodeLabelsMatchSpreadConstraints checks if ALL topology keys in spread Constraints are present in node labels.
func nodeLabelsMatchSpreadConstraints(nodeLabels map[string]string, constraints []topologySpreadConstraint) bool {
	for _, c := range constraints {
		if _, ok := nodeLabels[c.TopologyKey]; !ok {
			return false
		}
	}
	return true
}

func (pl *PodTopologySpread) filterTopologySpreadConstraints(constraints []v1.TopologySpreadConstraint, podLabels map[string]string, action v1.UnsatisfiableConstraintAction) ([]topologySpreadConstraint, error) {
	var result []topologySpreadConstraint
	for _, c := range constraints {
		if c.WhenUnsatisfiable == action {
			selector, err := metav1.LabelSelectorAsSelector(c.LabelSelector)
			if err != nil {
				return nil, err
			}

			if pl.enableMatchLabelKeysInPodTopologySpread && len(c.MatchLabelKeys) > 0 {
				matchLabels := make(labels.Set)
				for _, labelKey := range c.MatchLabelKeys {
					if value, ok := podLabels[labelKey]; ok {
						matchLabels[labelKey] = value
					}
				}
				if len(matchLabels) > 0 {
					selector = mergeLabelSetWithSelector(matchLabels, selector)
				}
			}

			tsc := topologySpreadConstraint{
				MaxSkew:            c.MaxSkew,
				TopologyKey:        c.TopologyKey,
				Selector:           selector,
				MinDomains:         ptr.Deref(c.MinDomains, 1),   // If MinDomains is nil, we treat MinDomains as 1.
				NodeAffinityPolicy: v1.NodeInclusionPolicyHonor,  // If NodeAffinityPolicy is nil, we treat NodeAffinityPolicy as "Honor".
				NodeTaintsPolicy:   v1.NodeInclusionPolicyIgnore, // If NodeTaintsPolicy is nil, we treat NodeTaintsPolicy as "Ignore".
			}
			if pl.enableNodeInclusionPolicyInPodTopologySpread {
				if c.NodeAffinityPolicy != nil {
					tsc.NodeAffinityPolicy = *c.NodeAffinityPolicy
				}
				if c.NodeTaintsPolicy != nil {
					tsc.NodeTaintsPolicy = *c.NodeTaintsPolicy
				}
			}
			result = append(result, tsc)
		}
	}
	return result, nil
}

func mergeLabelSetWithSelector(matchLabels labels.Set, s labels.Selector) labels.Selector {
	mergedSelector := labels.SelectorFromSet(matchLabels)

	requirements, ok := s.Requirements()
	if !ok {
		return s
	}

	for _, r := range requirements {
		mergedSelector = mergedSelector.Add(r)
	}

	return mergedSelector
}

func countPodsMatchSelector(podInfos []fwk.PodInfo, selector labels.Selector, ns string) int {
	if selector.Empty() {
		return 0
	}
	count := 0
	for _, p := range podInfos {
		// Bypass terminating Pod (see #87621).
		if p.GetPod().DeletionTimestamp != nil || p.GetPod().Namespace != ns {
			continue
		}
		if selector.Matches(labels.Set(p.GetPod().Labels)) {
			count++
		}
	}
	return count
}

// podLabelsMatchSpreadConstraints returns whether tha labels matches with the selector in any of topologySpreadConstraint
func podLabelsMatchSpreadConstraints(constraints []topologySpreadConstraint, labels labels.Set) bool {
	for _, c := range constraints {
		if c.Selector.Matches(labels) {
			return true
		}
	}
	return false
}
