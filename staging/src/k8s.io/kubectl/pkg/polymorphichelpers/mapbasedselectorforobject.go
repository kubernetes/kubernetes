/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"fmt"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

// mapBasedSelectorForObject returns the map-based selector associated with the provided object. If a
// new set-based selector is provided, an error is returned if the selector cannot be converted to a
// map-based selector
func mapBasedSelectorForObject(object runtime.Object) (string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *corev1.ReplicationController:
		return MakeLabels(t.Spec.Selector), nil

	case *corev1.Pod:
		if len(t.Labels) == 0 {
			return "", fmt.Errorf("the pod has no labels and cannot be exposed")
		}
		return MakeLabels(t.Labels), nil

	case *corev1.Service:
		if t.Spec.Selector == nil {
			return "", fmt.Errorf("the service has no pod selector set")
		}
		return MakeLabels(t.Spec.Selector), nil

	case *extensionsv1beta1.Deployment:
		// "extensions" deployments use pod template labels if selector is not set.
		var labels map[string]string
		if t.Spec.Selector != nil {
			// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
			// operator, DoubleEquals operator and In operator with only one element in the set.
			if len(t.Spec.Selector.MatchExpressions) > 0 {
				return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
			}
			labels = t.Spec.Selector.MatchLabels
		} else {
			labels = t.Spec.Template.Labels
		}
		if len(labels) == 0 {
			return "", fmt.Errorf("the deployment has no labels or selectors and cannot be exposed")
		}
		return MakeLabels(labels), nil

	case *appsv1.Deployment:
		// "apps" deployments must have the selector set.
		if t.Spec.Selector == nil || len(t.Spec.Selector.MatchLabels) == 0 {
			return "", fmt.Errorf("invalid deployment: no selectors, therefore cannot be exposed")
		}
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return MakeLabels(t.Spec.Selector.MatchLabels), nil

	case *appsv1beta2.Deployment:
		// "apps" deployments must have the selector set.
		if t.Spec.Selector == nil || len(t.Spec.Selector.MatchLabels) == 0 {
			return "", fmt.Errorf("invalid deployment: no selectors, therefore cannot be exposed")
		}
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return MakeLabels(t.Spec.Selector.MatchLabels), nil

	case *appsv1beta1.Deployment:
		// "apps" deployments must have the selector set.
		if t.Spec.Selector == nil || len(t.Spec.Selector.MatchLabels) == 0 {
			return "", fmt.Errorf("invalid deployment: no selectors, therefore cannot be exposed")
		}
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return MakeLabels(t.Spec.Selector.MatchLabels), nil

	case *extensionsv1beta1.ReplicaSet:
		// "extensions" replicasets use pod template labels if selector is not set.
		var labels map[string]string
		if t.Spec.Selector != nil {
			// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
			// operator, DoubleEquals operator and In operator with only one element in the set.
			if len(t.Spec.Selector.MatchExpressions) > 0 {
				return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
			}
			labels = t.Spec.Selector.MatchLabels
		} else {
			labels = t.Spec.Template.Labels
		}
		if len(labels) == 0 {
			return "", fmt.Errorf("the replica set has no labels or selectors and cannot be exposed")
		}
		return MakeLabels(labels), nil

	case *appsv1.ReplicaSet:
		// "apps" replicasets must have the selector set.
		if t.Spec.Selector == nil || len(t.Spec.Selector.MatchLabels) == 0 {
			return "", fmt.Errorf("invalid replicaset: no selectors, therefore cannot be exposed")
		}
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return MakeLabels(t.Spec.Selector.MatchLabels), nil

	case *appsv1beta2.ReplicaSet:
		// "apps" replicasets must have the selector set.
		if t.Spec.Selector == nil || len(t.Spec.Selector.MatchLabels) == 0 {
			return "", fmt.Errorf("invalid replicaset: no selectors, therefore cannot be exposed")
		}
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return MakeLabels(t.Spec.Selector.MatchLabels), nil

	default:
		return "", fmt.Errorf("cannot extract pod selector from %T", object)
	}

}

func MakeLabels(labels map[string]string) string {
	out := []string{}
	for key, value := range labels {
		out = append(out, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(out, ",")
}
