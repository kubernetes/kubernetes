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

package gvisor_reject

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type annotation struct {
	key, value string
}

var (
	// runtimeHandlerAnnotation is the gvisor pod annotation key/value supported by
	// GKE.
	runtimeHandlerAnnotation = annotation{
		key:   "runtime-handler.cri.kubernetes.io",
		value: "gvisor",
	}

	// untrustedWorkloadAnnotation is the gvisor pod annotation key/value
	// supported by containerd.
	untrustedWorkloadAnnotation = annotation{
		key:   "io.kubernetes.cri.untrusted-workload",
		value: "true",
	}

	gvisorToleration = api.Toleration{
		Key:      gvisorNodeKey,
		Operator: api.TolerationOpEqual,
		Value:    gvisorNodeValue,
		Effect:   api.TaintEffectNoSchedule,
	}

	gvisorNodeSelector = api.NodeSelectorRequirement{
		Key:      gvisorNodeKey,
		Operator: api.NodeSelectorOpIn,
		Values:   []string{gvisorNodeValue},
	}
)

const (
	// gvisorNodeKey is the key for gvisor node label and taint after beta.
	gvisorNodeKey = "sandbox.gke.io/runtime"

	// gvisorNodeValue is the value for gvisor node label and taint.
	gvisorNodeValue = "gvisor"

	// gvisorRuntimeClass is the name of the gvisor runtime class.
	gvisorRuntimeClass = "gvisor"
)

// PluginName indicates name of admission plugin.
const PluginName = "GvisorReject"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewGvisorReject(), nil
	})
}

// gvisorReject is an implementation of admission.Interface that rejects pods
// configured to run with gVisor when gVisor isn't enabled.
type gvisorReject struct {
	*admission.Handler
}

var _ admission.ValidationInterface = &gvisorReject{}

// NewGvisorReject creates a new gVisor admission control handler.
func NewGvisorReject() *gvisorReject {
	return &gvisorReject{
		// Update is not required for validation because:
		//   - Tolerations are append-only.
		//   - RuntimeClassName is immutable.
		//   - NodeAffinity is immutable.
		Handler: admission.NewHandler(admission.Create),
	}
}

func (g *gvisorReject) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) error {
	if !isPod(attributes) {
		return nil
	}
	if attributes.GetOperation() != admission.Create {
		return nil
	}

	pod, ok := attributes.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest(fmt.Sprintf("invalid object type: %T", attributes.GetObject()))
	}

	if err := validateGvisorPod(pod); err != nil {
		return admission.NewForbidden(attributes, fmt.Errorf("pod rejected: %v", err))
	}
	return nil
}

// ValidateGvisorPod checks that tolerations and node affinity are set to the
// pod spec so the pod is scheduled to gVisor node-pools appropriately.
func validateGvisorPod(pod *api.Pod) error {
	gvisorField := requestsGvisor(pod)
	if gvisorField == nil {
		// If no field selects for gVisor, this is not a gVisor pod.
		return nil
	}

	antiGvisorField := rejectsGvisor(pod)
	if antiGvisorField != nil {
		// If a field rejects gVisor, there is a conflict.
		return fmt.Errorf("conflict: %s requests gvisor, but %s rejects it", gvisorField.String(), antiGvisorField.String())
	}

	// This pod is set to run with gVisor. Validate that it has the right
	// toleration and node affinity set, otherwise it will be scheduled as a
	// regular pod.
	if !hasGvisorToleration(pod) || !hasGvisorNodeSelector(pod) {
		return errors.New("gVisor pod has invalid scheduling options. Check that you have at least one gVisor enabled node-pool")
	}
	return nil
}

func isPod(a admission.Attributes) bool {
	return a.GetResource().GroupResource() == api.Resource("pods") && len(a.GetSubresource()) == 0
}

// requestsGvisor checks whether the pod needs gVisor, and returns the field
// path for the selector indicating so. If gVisor is not required, returns nil.
func requestsGvisor(pod *api.Pod) *field.Path {
	// NOTE: This should really check the RuntimeHandler of the RuntimeClass, but
	// doing so introduces a potential TOCTOU race condition.
	if rc := pod.Spec.RuntimeClassName; rc != nil && *rc == gvisorRuntimeClass {
		return field.NewPath("spec", "runtimeClassName")
	}
	if pod.Annotations[runtimeHandlerAnnotation.key] == runtimeHandlerAnnotation.value {
		return field.NewPath("metadata", "annotations").Key(runtimeHandlerAnnotation.key)
	}
	if pod.Annotations[untrustedWorkloadAnnotation.key] == untrustedWorkloadAnnotation.value {
		return field.NewPath("metadata", "annotations").Key(untrustedWorkloadAnnotation.key)
	}
	return nil
}

// rejectsGvisor checks whether the pod explicitly rejects gVisor, and returns
// the field path for the selector indicating so. Returns nil otherwise.
func rejectsGvisor(pod *api.Pod) *field.Path {
	if rc := pod.Spec.RuntimeClassName; rc != nil && *rc != gvisorRuntimeClass && *rc != "" {
		return field.NewPath("spec", "runtimeClassName")
	}
	if val, ok := pod.Annotations[runtimeHandlerAnnotation.key]; ok && val != runtimeHandlerAnnotation.value {
		return field.NewPath("metadata", "annotations").Key(runtimeHandlerAnnotation.key)
	}
	if val, ok := pod.Annotations[untrustedWorkloadAnnotation.key]; ok && val != untrustedWorkloadAnnotation.value {
		return field.NewPath("metadata", "annotations").Key(untrustedWorkloadAnnotation.key)
	}
	return nil
}

// hasGvisorToleration checks that the following toleration exists:
//		tolerations:
//		- key: gke.sandbox.io/runtime
//			operator: Equal
//			value: gvisor
//			effect: NoSchedule
func hasGvisorToleration(pod *api.Pod) bool {
	for _, t := range pod.Spec.Tolerations {
		if t == gvisorToleration {
			return true
		}
	}
	return false
}

// hasGvisorNodeSelector checks that the following matchExpression exist in all
// requiredDuringSchedulingIgnoredDuringExecution matchExpressions:
//			affinity:
//				nodeAffinity:
//					requiredDuringSchedulingIgnoredDuringExecution:
//						nodeSelectorTerms:
//						- matchExpressions:
//							- key: gke.sandbox.io/runtime
//								operator: In
//								values:
//								- gvisor
func hasGvisorNodeSelector(pod *api.Pod) bool {
	if pod.Spec.Affinity == nil ||
		pod.Spec.Affinity.NodeAffinity == nil ||
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return false
	}

	selector := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if len(selector.NodeSelectorTerms) == 0 {
		return false
	}

	for _, term := range selector.NodeSelectorTerms {
		found := false
		for _, match := range term.MatchExpressions {
			if reflect.DeepEqual(match, gvisorNodeSelector) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
