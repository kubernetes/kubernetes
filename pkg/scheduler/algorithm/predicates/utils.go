/*
Copyright 2016 The Kubernetes Authors.

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

package predicates

import (
	"strings"

	"k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// FindLabelsInSet gets as many key/value pairs as possible out of a label set.
func FindLabelsInSet(labelsToKeep []string, selector labels.Set) map[string]string {
	aL := make(map[string]string)
	for _, l := range labelsToKeep {
		if selector.Has(l) {
			aL[l] = selector.Get(l)
		}
	}
	return aL
}

// AddUnsetLabelsToMap backfills missing values with values we find in a map.
func AddUnsetLabelsToMap(aL map[string]string, labelsToAdd []string, labelSet labels.Set) {
	for _, l := range labelsToAdd {
		// if the label is already there, dont overwrite it.
		if _, exists := aL[l]; exists {
			continue
		}
		// otherwise, backfill this label.
		if labelSet.Has(l) {
			aL[l] = labelSet.Get(l)
		}
	}
}

// FilterPodsByNamespace filters pods outside a namespace from the given list.
func FilterPodsByNamespace(pods []*v1.Pod, ns string) []*v1.Pod {
	filtered := []*v1.Pod{}
	for _, nsPod := range pods {
		if nsPod.Namespace == ns {
			filtered = append(filtered, nsPod)
		}
	}
	return filtered
}

// CreateSelectorFromLabels is used to define a selector that corresponds to the keys in a map.
func CreateSelectorFromLabels(aL map[string]string) labels.Selector {
	if aL == nil || len(aL) == 0 {
		return labels.Everything()
	}
	return labels.Set(aL).AsSelector()
}

// portsConflict check whether existingPorts and wantPorts conflict with each other
// return true if we have a conflict
func portsConflict(existingPorts schedulernodeinfo.HostPortInfo, wantPorts []*v1.ContainerPort) bool {
	for _, cp := range wantPorts {
		if existingPorts.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
			return true
		}
	}

	return false
}

// SetPredicatesOrderingDuringTest sets the predicatesOrdering to the specified
// value, and returns a function that restores the original value.
func SetPredicatesOrderingDuringTest(value []string) func() {
	origVal := predicatesOrdering
	predicatesOrdering = value
	return func() {
		predicatesOrdering = origVal
	}
}

// isCSIMigrationOn returns a boolean value indicating whether
// the CSI migration has been enabled for a particular storage plugin.
func isCSIMigrationOn(csiNode *storagev1beta1.CSINode, pluginName string) bool {
	if csiNode == nil || len(pluginName) == 0 {
		return false
	}

	// In-tree storage to CSI driver migration feature should be enabled,
	// along with the plugin-specific one
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationAWS) {
			return false
		}
	case csilibplugins.GCEPDInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationGCE) {
			return false
		}
	case csilibplugins.AzureDiskInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationAzureDisk) {
			return false
		}
	case csilibplugins.CinderInTreePluginName:
		if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigrationOpenStack) {
			return false
		}
	default:
		return false
	}

	// The plugin name should be listed in the CSINode object annotation.
	// This indicates that the plugin has been migrated to a CSI driver in the node.
	csiNodeAnn := csiNode.GetAnnotations()
	if csiNodeAnn == nil {
		return false
	}

	var mpaSet sets.String
	mpa := csiNodeAnn[v1.MigratedPluginsAnnotationKey]
	if len(mpa) == 0 {
		mpaSet = sets.NewString()
	} else {
		tok := strings.Split(mpa, ",")
		mpaSet = sets.NewString(tok...)
	}

	return mpaSet.Has(pluginName)
}

// utilities for building pod/node objects using a "chained" manner
type nodeSelectorWrapper struct{ v1.NodeSelector }

func makeNodeSelector() *nodeSelectorWrapper {
	return &nodeSelectorWrapper{v1.NodeSelector{}}
}

// NOTE: each time we append a selectorTerm into `s`
// and overall all selecterTerms are ORed
func (s *nodeSelectorWrapper) in(key string, vals []string) *nodeSelectorWrapper {
	expression := v1.NodeSelectorRequirement{
		Key:      key,
		Operator: v1.NodeSelectorOpIn,
		Values:   vals,
	}
	selectorTerm := v1.NodeSelectorTerm{}
	selectorTerm.MatchExpressions = append(selectorTerm.MatchExpressions, expression)
	s.NodeSelectorTerms = append(s.NodeSelectorTerms, selectorTerm)
	return s
}

func (s *nodeSelectorWrapper) obj() *v1.NodeSelector {
	return &s.NodeSelector
}

type labelSelectorWrapper struct{ metav1.LabelSelector }

func makeLabelSelector() *labelSelectorWrapper {
	return &labelSelectorWrapper{metav1.LabelSelector{}}
}

func (s *labelSelectorWrapper) label(k, v string) *labelSelectorWrapper {
	if s.MatchLabels == nil {
		s.MatchLabels = make(map[string]string)
	}
	s.MatchLabels[k] = v
	return s
}

func (s *labelSelectorWrapper) in(key string, vals []string) *labelSelectorWrapper {
	expression := metav1.LabelSelectorRequirement{
		Key:      key,
		Operator: metav1.LabelSelectorOpIn,
		Values:   vals,
	}
	s.MatchExpressions = append(s.MatchExpressions, expression)
	return s
}

func (s *labelSelectorWrapper) notIn(key string, vals []string) *labelSelectorWrapper {
	expression := metav1.LabelSelectorRequirement{
		Key:      key,
		Operator: metav1.LabelSelectorOpNotIn,
		Values:   vals,
	}
	s.MatchExpressions = append(s.MatchExpressions, expression)
	return s
}

func (s *labelSelectorWrapper) exists(k string) *labelSelectorWrapper {
	expression := metav1.LabelSelectorRequirement{
		Key:      k,
		Operator: metav1.LabelSelectorOpExists,
	}
	s.MatchExpressions = append(s.MatchExpressions, expression)
	return s
}

func (s *labelSelectorWrapper) notExist(k string) *labelSelectorWrapper {
	expression := metav1.LabelSelectorRequirement{
		Key:      k,
		Operator: metav1.LabelSelectorOpDoesNotExist,
	}
	s.MatchExpressions = append(s.MatchExpressions, expression)
	return s
}

func (s *labelSelectorWrapper) obj() *metav1.LabelSelector {
	return &s.LabelSelector
}

type podWrapper struct{ v1.Pod }

func makePod() *podWrapper {
	return &podWrapper{v1.Pod{}}
}

func (p *podWrapper) obj() *v1.Pod {
	return &p.Pod
}

func (p *podWrapper) name(s string) *podWrapper {
	p.Name = s
	return p
}

func (p *podWrapper) namespace(s string) *podWrapper {
	p.Namespace = s
	return p
}

func (p *podWrapper) node(s string) *podWrapper {
	p.Spec.NodeName = s
	return p
}

func (p *podWrapper) nodeSelector(m map[string]string) *podWrapper {
	p.Spec.NodeSelector = m
	return p
}

// particular represents HARD node affinity
func (p *podWrapper) nodeAffinityIn(key string, vals []string) *podWrapper {
	if p.Spec.Affinity == nil {
		p.Spec.Affinity = &v1.Affinity{}
	}
	if p.Spec.Affinity.NodeAffinity == nil {
		p.Spec.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	nodeSelector := makeNodeSelector().in(key, vals).obj()
	p.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = nodeSelector
	return p
}

func (p *podWrapper) spreadConstraint(maxSkew int, tpKey string, mode v1.UnsatisfiableConstraintResponse, selector *metav1.LabelSelector) *podWrapper {
	c := v1.TopologySpreadConstraint{
		MaxSkew:           int32(maxSkew),
		TopologyKey:       tpKey,
		WhenUnsatisfiable: mode,
		LabelSelector:     selector,
	}
	p.Spec.TopologySpreadConstraints = append(p.Spec.TopologySpreadConstraints, c)
	return p
}

func (p *podWrapper) label(k, v string) *podWrapper {
	if p.Labels == nil {
		p.Labels = make(map[string]string)
	}
	p.Labels[k] = v
	return p
}

type nodeWrapper struct{ v1.Node }

func makeNode() *nodeWrapper {
	return &nodeWrapper{v1.Node{}}
}

func (n *nodeWrapper) obj() *v1.Node {
	return &n.Node
}

func (n *nodeWrapper) name(s string) *nodeWrapper {
	n.Name = s
	return n
}

func (n *nodeWrapper) label(k, v string) *nodeWrapper {
	if n.Labels == nil {
		n.Labels = make(map[string]string)
	}
	n.Labels[k] = v
	return n
}
