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

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	corev1 "k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl"
)

// mapBasedSelectorForObject returns the map-based selector associated with the provided object. If a
// new set-based selector is provided, an error is returned if the selector cannot be converted to a
// map-based selector
func mapBasedSelectorForObject(object runtime.Object) (string, error) {
	// TODO: replace with a swagger schema based approach (identify pod selector via schema introspection)
	switch t := object.(type) {
	case *corev1.ReplicationController:
		return kubectl.MakeLabels(t.Spec.Selector), nil

	case *corev1.Pod:
		if len(t.Labels) == 0 {
			return "", fmt.Errorf("the pod has no labels and cannot be exposed")
		}
		return kubectl.MakeLabels(t.Labels), nil

	case *corev1.Service:
		if t.Spec.Selector == nil {
			return "", fmt.Errorf("the service has no pod selector set")
		}
		return kubectl.MakeLabels(t.Spec.Selector), nil

	case *extensionsv1beta1.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *appsv1.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *appsv1beta2.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *appsv1beta1.Deployment:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil

	case *extensionsv1beta1.ReplicaSet:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *appsv1.ReplicaSet:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil
	case *appsv1beta2.ReplicaSet:
		// TODO(madhusudancs): Make this smarter by admitting MatchExpressions with Equals
		// operator, DoubleEquals operator and In operator with only one element in the set.
		if len(t.Spec.Selector.MatchExpressions) > 0 {
			return "", fmt.Errorf("couldn't convert expressions - \"%+v\" to map-based selector format", t.Spec.Selector.MatchExpressions)
		}
		return kubectl.MakeLabels(t.Spec.Selector.MatchLabels), nil

	default:
		return "", fmt.Errorf("cannot extract pod selector from %T", object)
	}
}
