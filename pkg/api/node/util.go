/*
Copyright 2022 The Kubernetes Authors.

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

package node

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
)

var deprecatedNodeLabels = map[string]string{
	`beta.kubernetes.io/arch`:                  `deprecated since v1.14; use "kubernetes.io/arch" instead`,
	`beta.kubernetes.io/os`:                    `deprecated since v1.14; use "kubernetes.io/os" instead`,
	`failure-domain.beta.kubernetes.io/region`: `deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
	`failure-domain.beta.kubernetes.io/zone`:   `deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
	`beta.kubernetes.io/instance-type`:         `deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
}

// GetNodeLabelDeprecatedMessage returns the message for the deprecated node label
// and a bool indicating if the label is deprecated.
func GetNodeLabelDeprecatedMessage(key string) (string, bool) {
	msg, ok := deprecatedNodeLabels[key]
	return msg, ok
}

func GetWarningsForRuntimeClass(rc *node.RuntimeClass) []string {
	var warnings []string

	if rc != nil && rc.Scheduling != nil && rc.Scheduling.NodeSelector != nil {
		// use of deprecated node labels in scheduling's node affinity
		for key := range rc.Scheduling.NodeSelector {
			if msg, deprecated := GetNodeLabelDeprecatedMessage(key); deprecated {
				warnings = append(warnings, fmt.Sprintf("%s: %s", field.NewPath("scheduling", "nodeSelector"), msg))
			}
		}
	}

	return warnings
}

// GetWarningsForNodeSelector tests if any of the node selector requirements in the template is deprecated.
// If there are deprecated node selector requirements in either match expressions or match labels, a warning is returned.
func GetWarningsForNodeSelector(nodeSelector *metav1.LabelSelector, fieldPath *field.Path) []string {
	if nodeSelector == nil {
		return nil
	}

	var warnings []string
	// use of deprecated node labels in matchLabelExpressions
	for i, expression := range nodeSelector.MatchExpressions {
		if msg, deprecated := GetNodeLabelDeprecatedMessage(expression.Key); deprecated {
			warnings = append(
				warnings,
				fmt.Sprintf(
					"%s: %s is %s",
					fieldPath.Child("matchExpressions").Index(i).Child("key"),
					expression.Key,
					msg,
				),
			)
		}
	}

	// use of deprecated node labels in matchLabels
	for label := range nodeSelector.MatchLabels {
		if msg, deprecated := GetNodeLabelDeprecatedMessage(label); deprecated {
			warnings = append(warnings, fmt.Sprintf("%s: %s", fieldPath.Child("matchLabels").Child(label), msg))
		}
	}
	return warnings
}

// GetWarningsForNodeSelectorTerm checks match expressions of node selector term
func GetWarningsForNodeSelectorTerm(nodeSelectorTerm api.NodeSelectorTerm, fieldPath *field.Path) []string {
	var warnings []string
	// use of deprecated node labels in matchLabelExpressions
	for i, expression := range nodeSelectorTerm.MatchExpressions {
		if msg, deprecated := GetNodeLabelDeprecatedMessage(expression.Key); deprecated {
			warnings = append(
				warnings,
				fmt.Sprintf(
					"%s: %s is %s",
					fieldPath.Child("matchExpressions").Index(i).Child("key"),
					expression.Key,
					msg,
				),
			)
		}
	}
	return warnings
}
