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

// Package podrestoreauthorization contains an admission plugin that gates Pod
// restore on a dedicated authorization check. Restoring a Pod from a
// PodCheckpoint consumes the checkpoint's captured process and memory state and
// is more sensitive than merely reading the PodCheckpoint object. Reading the
// object is ordinary RBAC (get/list/watch on podcheckpoints); restoring is
// authorized separately via the "restore" verb on podcheckpoints.
package podrestoreauthorization

import (
	"context"
	"fmt"
	"io"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	checkpointutil "k8s.io/kubernetes/pkg/apis/checkpoint/util"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName is the name of this admission plugin.
const PluginName = "PodRestoreAuthorization"

// checkpointGroup is the API group of the PodCheckpoint resource that the
// "restore" verb is authorized against.
const checkpointGroup = "checkpoint.k8s.io"

// podCheckpointGVR is the resource used to resolve a pod's
// spec.restoreFrom.name to the PodCheckpoint it restores from.
var podCheckpointGVR = schema.GroupVersionResource{
	Group:    checkpointGroup,
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// Register registers the plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newPlugin(), nil
	})
}

// Plugin authorizes, mutates, and validates Pod restore. When a Pod is created
// with spec.restoreFrom set:
//
//   - Admit (mutating) injects a required node affinity pinning the Pod to the
//     node recorded in the referenced checkpoint, so the scheduler places the Pod
//     on the node that holds the checkpoint data. The restore goes through the
//     scheduler rather than bypassing it; the plugin no longer sets spec.nodeName.
//     Existing required node affinity is preserved, with the node-name constraint
//     added to every selector term so the constraints retain their AND/OR meaning.
//     A restore Pod that supplies spec.nodeName is rejected because placement is
//     admission-controlled.
//   - Validate (validating) requires the requester to be authorized for the
//     "restore" verb on the referenced PodCheckpoint in the Pod's namespace, and
//     requires the Pod's spec to equal the pod template captured in the
//     checkpoint. The equality check normalizes the injected node constraint
//     (along with spec.nodeName and spec.restoreFrom). The equality check here is
//     authoritative; the kubelet re-checks it before the CRI restore as defense
//     in depth.
type Plugin struct {
	*admission.Handler
	authz         authorizer.UnconditionalAuthorizer
	dynamicClient dynamic.Interface
}

var (
	_ admission.MutationInterface                              = &Plugin{}
	_ admission.ValidationInterface                            = &Plugin{}
	_ genericadmissioninitializer.WantsUnconditionalAuthorizer = &Plugin{}
	_ genericadmissioninitializer.WantsDynamicClient           = &Plugin{}
)

func newPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

// SetUnconditionalAuthorizer sets the authorizer used to issue the restore
// authorization check.
func (p *Plugin) SetUnconditionalAuthorizer(a authorizer.UnconditionalAuthorizer) {
	p.authz = a
}

// SetDynamicClient sets the client used to read the referenced PodCheckpoint
// for the spec-equality check.
func (p *Plugin) SetDynamicClient(c dynamic.Interface) {
	p.dynamicClient = c
}

// ValidateInitialization ensures the required dependencies were injected.
func (p *Plugin) ValidateInitialization() error {
	if p.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if p.dynamicClient == nil {
		return fmt.Errorf("%s requires a dynamic client", PluginName)
	}
	return nil
}

var podResource = api.Resource("pods")

// restoringPod returns the incoming Pod and checkpoint name when this request
// creates a Pod with spec.restoreFrom set. restoreFrom is immutable, so updates
// never start another restore and need no admission processing. The gating is
// identical for Admit and Validate.
func restoringPod(a admission.Attributes) (pod *api.Pod, checkpointName string, ok bool) {
	// The feature gate governs the whole Pod-level checkpoint/restore feature;
	// when it is off, spec.restoreFrom is already rejected by validation and
	// there is nothing to do.
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) {
		return nil, "", false
	}

	// Only act on Pod creation, not updates or subresources. Core validation
	// enforces restoreFrom immutability after creation.
	if a.GetOperation() != admission.Create || a.GetResource().GroupResource() != podResource || a.GetSubresource() != "" {
		return nil, "", false
	}

	pod, isPod := a.GetObject().(*api.Pod)
	if !isPod {
		// Not a Pod object (e.g. a DeleteOptions); nothing to do.
		return nil, "", false
	}

	// Nothing to do unless a restore is requested.
	if pod.Spec.RestoreFrom == nil || pod.Spec.RestoreFrom.Name == "" {
		return nil, "", false
	}

	return pod, pod.Spec.RestoreFrom.Name, true
}

// Admit injects a required node affinity pinning the restoring Pod to the node
// recorded in the referenced checkpoint, so the scheduler places it on the node
// that holds the checkpoint data. Existing required affinity terms are retained;
// adding the node requirement to each term preserves their OR-between-terms and
// AND-within-a-term semantics.
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	pod, checkpointName, ok := restoringPod(a)
	if !ok {
		return nil
	}

	obj, err := p.dynamicClient.Resource(podCheckpointGVR).Namespace(a.GetNamespace()).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	checkpointNode, _, err := unstructured.NestedString(obj.Object, "status", "nodeName")
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read status.nodeName from PodCheckpoint %q: %w", checkpointName, err))
	}
	if checkpointNode == "" {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has not recorded status.nodeName; restore cannot be scheduled until checkpointing has started on a node", checkpointName))
	}

	// Restore placement is admission-controlled: the Pod is pinned to the
	// checkpoint's node via an injected node affinity. A direct node binding would
	// bypass the scheduler, so reject it rather than silently replacing it.
	if pod.Spec.NodeName != "" {
		return admission.NewForbidden(a, fmt.Errorf("pod restoring from PodCheckpoint %q must not set spec.nodeName; restore placement is admission-controlled and pins the Pod to node %q via node affinity", checkpointName, checkpointNode))
	}

	// Inject the required node affinity selecting the checkpoint's node by name,
	// preserving preferred node affinity and pod/anti-affinity.
	if pod.Spec.Affinity == nil {
		pod.Spec.Affinity = &api.Affinity{}
	}
	if pod.Spec.Affinity.NodeAffinity == nil {
		pod.Spec.Affinity.NodeAffinity = &api.NodeAffinity{}
	}
	nodeAffinity := pod.Spec.Affinity.NodeAffinity
	required := nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if required == nil || len(required.NodeSelectorTerms) == 0 {
		required = &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{{}}}
		nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = required
	}
	for i := range required.NodeSelectorTerms {
		required.NodeSelectorTerms[i].MatchFields = append(required.NodeSelectorTerms[i].MatchFields, api.NodeSelectorRequirement{
			Key:      "metadata.name",
			Operator: api.NodeSelectorOpIn,
			Values:   []string{checkpointNode},
		})
	}
	return nil
}

// Validate authorizes a newly-created restore Pod and enforces that its spec
// equals the pod template captured in the checkpoint.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	pod, checkpointName, ok := restoringPod(a)
	if !ok {
		return nil
	}

	attrs := authorizer.AttributesRecord{
		User:            a.GetUserInfo(),
		Verb:            "restore",
		APIGroup:        checkpointGroup,
		APIVersion:      "v1alpha1",
		Resource:        "podcheckpoints",
		Name:            checkpointName,
		Namespace:       a.GetNamespace(),
		ResourceRequest: true,
	}

	decision, reason, err := p.authz.Authorize(ctx, attrs)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("error authorizing restore from PodCheckpoint %q: %w", checkpointName, err))
	}
	if decision != authorizer.DecisionAllow {
		return admission.NewForbidden(a, fmt.Errorf("pod sets spec.restoreFrom=%q but the requester is not authorized to restore it (requires the %q verb on podcheckpoints in namespace %q): %s", checkpointName, "restore", a.GetNamespace(), reason))
	}

	// The requester is authorized to restore. Read the referenced checkpoint and
	// enforce that the Pod's spec equals the captured pod template. The kubelet
	// re-checks equality before the CRI restore as defense in depth.
	obj, err := p.dynamicClient.Resource(podCheckpointGVR).Namespace(a.GetNamespace()).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	return validatePodSpecMatchesCheckpoint(o.GetObjectConvertor(), a, pod, checkpointName, obj)
}

// validatePodSpecMatchesCheckpoint enforces that the restoring Pod's spec equals
// the pod template captured in the checkpoint. The restored process tree depends
// on the spec it was checkpointed with (resources, mounts, security context,
// containers), so the spec must not change between checkpoint and restore.
//
// The comparison runs after API defaulting (both sides are defaulted) and uses
// the same sanitization as checkpoint capture and the kubelet's defense-in-depth
// check. That normalization removes spec.nodeName, spec.restoreFrom, and only
// node-identity scheduling constraints, including the metadata.name requirement
// Admit injects. Other required node affinity remains part of the authoritative
// template and must match.
//
// The check is skipped when the template is not yet populated. Admit still
// requires status.nodeName so the Pod can be pinned safely; the kubelet keeps a
// not-yet-Ready restore Pending and performs the equality check before CRI restore.
func validatePodSpecMatchesCheckpoint(convertor runtime.ObjectConvertor, a admission.Attributes, pod *api.Pod, checkpointName string, obj *unstructured.Unstructured) error {
	tmplRaw, found, err := unstructured.NestedMap(obj.Object, "status", "checkpointedPodTemplate")
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read status.checkpointedPodTemplate from PodCheckpoint %q: %w", checkpointName, err))
	}
	if !found || len(tmplRaw) == 0 {
		return nil
	}

	var tmpl v1.PodTemplateSpec
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(tmplRaw, &tmpl); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to decode status.checkpointedPodTemplate from PodCheckpoint %q: %w", checkpointName, err))
	}

	// Convert the incoming (internal) Pod to v1 so it compares on equal footing
	// with the stored v1 template, using the convertor admission provides.
	var live v1.Pod
	if err := convertor.Convert(pod, &live, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert pod for the PodCheckpoint %q equality check: %w", checkpointName, err))
	}

	want := tmpl.Spec.DeepCopy()
	got := checkpointutil.SanitizePodTemplate(&live).Spec.DeepCopy()

	if !equality.Semantic.DeepEqual(*want, *got) {
		errs := field.ErrorList{field.Forbidden(
			field.NewPath("spec"),
			fmt.Sprintf("must match the pod spec captured in PodCheckpoint %q; changing the spec between checkpoint and restore is not permitted (spec.nodeName, spec.restoreFrom, and node-local placement constraints excepted)", checkpointName),
		)}
		return apierrors.NewInvalid(schema.GroupKind{Kind: "Pod"}, pod.Name, errs)
	}
	return nil
}
