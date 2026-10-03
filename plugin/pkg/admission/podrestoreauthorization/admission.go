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
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName is the name of this admission plugin.
const PluginName = "PodRestoreAuthorization"

// checkpointGroup is the API group of the PodCheckpoint resource that the
// "restore" verb is authorized against.
const checkpointGroup = "node.k8s.io"

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
//     added to every nonempty term so the constraints retain their AND/OR meaning.
//     A restore Pod that supplies spec.nodeName is rejected because placement is
//     admission-controlled.
//   - Validate (validating) requires the requester to be authorized for the
//     "restore" verb on the referenced PodCheckpoint in the Pod's namespace, and
//     requires a Ready checkpoint with a captured pod template, and requires
//     the Pod's spec to equal the pod template captured in the
//     checkpoint with the same node pin added to the expected spec.
//     The equality check here is authoritative; the kubelet re-checks it before
//     the CRI restore as defense in depth.
type Plugin struct {
	*admission.Handler
	authz  authorizer.UnconditionalAuthorizer
	client kubernetes.Interface
}

var (
	_ admission.MutationInterface                              = &Plugin{}
	_ admission.ValidationInterface                            = &Plugin{}
	_ genericadmissioninitializer.WantsUnconditionalAuthorizer = &Plugin{}
	_ genericadmissioninitializer.WantsExternalKubeClientSet   = &Plugin{}
)

func newPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// SetUnconditionalAuthorizer sets the authorizer used to issue the restore
// authorization check.
func (p *Plugin) SetUnconditionalAuthorizer(a authorizer.UnconditionalAuthorizer) {
	p.authz = a
}

// SetExternalKubeClientSet sets the client used to read the referenced PodCheckpoint
// for the spec-equality check.
func (p *Plugin) SetExternalKubeClientSet(c kubernetes.Interface) {
	p.client = c
}

// ValidateInitialization ensures the required dependencies were injected.
func (p *Plugin) ValidateInitialization() error {
	if p.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if p.client == nil {
		return fmt.Errorf("%s requires a Kubernetes client", PluginName)
	}
	return nil
}

var podResource = api.Resource("pods")

// restoringPod returns the incoming Pod and checkpoint name when this request
// creates a Pod with spec.restoreFrom set. Updates are checked separately to
// preserve the admitted workload until restore succeeds.
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

	checkpoint, err := p.client.NodeV1alpha1().PodCheckpoints(a.GetNamespace()).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	if checkpoint.Status.NodeName == nil || *checkpoint.Status.NodeName == "" {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has not recorded status.nodeName; restore cannot be scheduled until checkpointing has started on a node", checkpointName))
	}
	checkpointNode := *checkpoint.Status.NodeName

	// Restore placement is admission-controlled: the Pod is pinned to the
	// checkpoint's node via an injected node affinity. A direct node binding would
	// bypass the scheduler, so reject it rather than silently replacing it.
	if pod.Spec.NodeName != "" {
		return admission.NewForbidden(a, fmt.Errorf("pod restoring from PodCheckpoint %q must not set spec.nodeName; restore placement is admission-controlled and pins the Pod to node %q via node affinity", checkpointName, checkpointNode))
	}

	// Use the same transformation as the comparison against the captured spec.
	var live v1.Pod
	if err := o.GetObjectConvertor().Convert(pod, &live, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert restore Pod: %w", err))
	}
	// Conversion can share pointers with the admission object.
	spec := live.Spec.DeepCopy()
	checkpointutil.AddRestoreNodeAffinity(spec, checkpointNode)
	var affinity api.Affinity
	if err := corev1.Convert_v1_Affinity_To_core_Affinity(spec.Affinity, &affinity, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert restore node affinity: %w", err))
	}
	pod.Spec.Affinity = &affinity
	return nil
}

// Validate authorizes a newly-created restore Pod and enforces that its spec
// equals the pod template captured in the checkpoint. It also prevents workload
// updates before restore succeeds.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetOperation() == admission.Update {
		return validateRestoreUpdate(a)
	}
	pod, checkpointName, ok := restoringPod(a)
	if !ok {
		return nil
	}

	// A later mutating webhook can set nodeName after Admit checked it.
	// Enforce scheduler placement again on the final create request.
	if pod.Spec.NodeName != "" {
		return admission.NewForbidden(a, fmt.Errorf("pod restoring from PodCheckpoint %q must not set spec.nodeName; restore placement is admission-controlled via node affinity", checkpointName))
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
	checkpoint, err := p.client.NodeV1alpha1().PodCheckpoints(a.GetNamespace()).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	if !apimeta.IsStatusConditionTrue(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady) {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q is not ready; restore requires status.conditions Ready=True", checkpointName))
	}
	// Validate the final request after mutating admission, including any options
	// inserted by a later webhook. Restore options use their own allowlist.
	if len(pod.Spec.RestoreFrom.Options) != 0 {
		if pod.Spec.RuntimeClassName == nil || *pod.Spec.RuntimeClassName == "" {
			return admission.NewForbidden(a, fmt.Errorf("spec.restoreFrom.options requires spec.runtimeClassName and a RuntimeClass restore option allowlist"))
		}
		className := *pod.Spec.RuntimeClassName
		class, err := p.client.NodeV1().RuntimeClasses().Get(ctx, className, metav1.GetOptions{})
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("cannot read RuntimeClass %q for spec.restoreFrom.options: %w", className, err))
		}
		var allowed []string
		if class.PodCheckpoint != nil {
			allowed = class.PodCheckpoint.AllowedRestoreOptions
		}
		if err := checkpointutil.ValidateRuntimeOptions(pod.Spec.RestoreFrom.Options, allowed); err != nil {
			return admission.NewForbidden(a, fmt.Errorf("spec.restoreFrom.options for RuntimeClass %q: %w", className, err))
		}
	}
	return validatePodSpecMatchesCheckpoint(o.GetObjectConvertor(), a, pod, checkpointName, checkpoint)
}

// validateRestoreUpdate preserves the admitted workload until restore succeeds.
// Checking the old status prevents a spec update from claiming restore success.
// Status and binding updates have their own authorization and validation.
func validateRestoreUpdate(a admission.Attributes) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) || a.GetResource().GroupResource() != podResource {
		return nil
	}
	switch a.GetSubresource() {
	case "", "resize", "ephemeralcontainers":
	default:
		return nil
	}
	pod, ok := a.GetObject().(*api.Pod)
	oldPod, oldOK := a.GetOldObject().(*api.Pod)
	if !ok || !oldOK || oldPod.Spec.RestoreFrom == nil {
		return nil
	}
	// Core validation rejects changing restoreFrom independently.
	if !equality.Semantic.DeepEqual(oldPod.Spec.RestoreFrom, pod.Spec.RestoreFrom) {
		return nil
	}
	for _, condition := range oldPod.Status.Conditions {
		if condition.Type == api.PodRestored && condition.Status == api.ConditionTrue {
			return nil
		}
	}
	want, got := oldPod.Spec.DeepCopy(), pod.Spec.DeepCopy()
	// Removing a gate must remain possible so a restore can be scheduled.
	want.SchedulingGates, got.SchedulingGates = nil, nil
	if !equality.Semantic.DeepEqual(want, got) {
		return admission.NewForbidden(a, fmt.Errorf("pod spec cannot change until restore succeeds (PodRestored=True); scheduling gates may be removed"))
	}
	return nil
}

// validatePodSpecMatchesCheckpoint enforces that the restoring Pod's spec equals
// the pod template captured in the checkpoint. The restored process tree depends
// on the spec it was checkpointed with (resources, mounts, security context,
// containers), so the spec must not change between checkpoint and restore.
//
// The comparison preserves the captured scheduling constraints and adds the
// same checkpoint-node pin as Admit to the expected spec. Only nodeName,
// restoreFrom, ephemeral containers, and scheduling gates are excluded by
// sanitization; user-supplied node identities remain part of the comparison.
//
// A captured template is required so admission can verify the workload before
// accepting a restore Pod.
func validatePodSpecMatchesCheckpoint(convertor runtime.ObjectConvertor, a admission.Attributes, pod *api.Pod, checkpointName string, checkpoint *nodev1alpha1.PodCheckpoint) error {
	tmpl := checkpoint.Status.CheckpointedPodTemplate
	if tmpl == nil {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has no status.checkpointedPodTemplate; restore requires a captured pod template", checkpointName))
	}

	// Convert the incoming (internal) Pod to v1 so it compares on equal footing
	// with the stored v1 template, using the convertor admission provides.
	var live v1.Pod
	if err := convertor.Convert(pod, &live, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert pod for the PodCheckpoint %q equality check: %w", checkpointName, err))
	}

	if checkpoint.Status.NodeName == nil || *checkpoint.Status.NodeName == "" {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has no status.nodeName", checkpointName))
	}
	want := tmpl.Spec.DeepCopy()
	checkpointutil.AddRestoreNodeAffinity(want, *checkpoint.Status.NodeName)
	got := checkpointutil.SanitizePodTemplate(&live).Spec.DeepCopy()

	if !equality.Semantic.DeepEqual(*want, *got) {
		errs := field.ErrorList{field.Forbidden(
			field.NewPath("spec"),
			fmt.Sprintf("must match the pod spec captured in PodCheckpoint %q; changing the spec between checkpoint and restore is not permitted (spec.nodeName, restoreFrom, ephemeral containers, and scheduling gates excepted; the checkpoint-node affinity pin is added by admission)", checkpointName),
		)}
		return apierrors.NewInvalid(schema.GroupKind{Kind: "Pod"}, pod.Name, errs)
	}
	return nil
}
