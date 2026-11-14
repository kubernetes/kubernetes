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

// Package podrestoreauthorization prepares restore Pods for scheduling and
// runtime cleanup. It authorizes restore before reading checkpoint state for
// mutation. Pod REST storage independently authorizes and validates the final
// object, including when this plugin is disabled.
package podrestoreauthorization

import (
	"context"
	"fmt"
	"io"
	"slices"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
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

// Plugin injects the checkpoint-node affinity and restore protection finalizer
// when a restore Pod is created. It authorizes the requester before reading the
// checkpoint so mutation cannot disclose checkpoint state to an unauthorized
// caller. Pod REST storage enforces authorization and restore constraints after
// all mutating admission has completed.
type Plugin struct {
	*admission.Handler
	authz  authorizer.UnconditionalAuthorizer
	client kubernetes.Interface
}

var (
	_ admission.MutationInterface                              = &Plugin{}
	_ genericadmissioninitializer.WantsUnconditionalAuthorizer = &Plugin{}
	_ genericadmissioninitializer.WantsExternalKubeClientSet   = &Plugin{}
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

// SetExternalKubeClientSet sets the client used to read the referenced PodCheckpoint
// for placement mutation.
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
// creates a Pod with spec.restoreFrom set. Pod validation preserves the admitted
// workload until restore succeeds.
func restoringPod(a admission.Attributes) (pod *api.Pod, checkpointName string, ok bool) {
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

func (p *Plugin) authorizeRestore(ctx context.Context, a admission.Attributes, checkpointName string) error {
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

	return nil
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

	// The checkpoint read and placement errors must not expose its state to
	// someone who is not authorized to restore it.
	if err := p.authorizeRestore(ctx, a, checkpointName); err != nil {
		return err
	}

	checkpoint, err := p.client.NodeV1alpha1().PodCheckpoints(a.GetNamespace()).Get(ctx, checkpointName, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to read PodCheckpoint %q referenced by spec.restoreFrom: %w", checkpointName, err))
	}
	if checkpoint.Status.NodeName == nil || *checkpoint.Status.NodeName == "" {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has no status.nodeName assignment; restore cannot be scheduled until the checkpoint controller assigns a node", checkpointName))
	}
	if checkpoint.DeletionTimestamp != nil {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q is being deleted", checkpointName))
	}
	if pod.Spec.RestoreFrom.UID != "" && pod.Spec.RestoreFrom.UID != checkpoint.UID {
		return admission.NewForbidden(a, fmt.Errorf("PodCheckpoint %q has UID %q, not requested UID %q", checkpointName, checkpoint.UID, pod.Spec.RestoreFrom.UID))
	}
	pod.Spec.RestoreFrom.UID = checkpoint.UID
	checkpointNode := *checkpoint.Status.NodeName

	// Restore placement uses the scheduler: the Pod is pinned to the
	// checkpoint's node via an injected node affinity. A direct node binding would
	// bypass the scheduler, so reject it rather than silently replacing it.
	if pod.Spec.NodeName != "" {
		return admission.NewForbidden(a, fmt.Errorf("pod restoring from PodCheckpoint %q must not set spec.nodeName; restore placement pins the Pod to node %q via node affinity", checkpointName, checkpointNode))
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
	if !slices.Contains(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer) {
		pod.Finalizers = append(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	}
	return nil
}
