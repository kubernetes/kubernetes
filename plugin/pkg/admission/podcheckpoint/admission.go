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

// Package podcheckpoint validates source Pods before checkpoint creation and
// preserves their workload while a checkpoint captures it.
package podcheckpoint

import (
	"context"
	"fmt"
	"io"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	initializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	api "k8s.io/kubernetes/pkg/apis/core"
	node "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName is the name of the checkpoint admission plugin.
const PluginName = "PodCheckpoint"

// Register registers the plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(io.Reader) (admission.Interface, error) {
		return &Plugin{Handler: admission.NewHandler(admission.Create, admission.Update)}, nil
	})
}

// Plugin requires a running source Pod before accepting a checkpoint request and
// rejects workload changes while the kubelet captures that Pod.
type Plugin struct {
	*admission.Handler
	client                kubernetes.Interface
	enabled               bool
	inspectedFeatureGates bool
}

var (
	_ admission.ValidationInterface          = &Plugin{}
	_ initializer.WantsExternalKubeClientSet = &Plugin{}
	_ initializer.WantsFeatures              = &Plugin{}
)

// SetExternalKubeClientSet supplies the client used to read source Pods.
func (p *Plugin) SetExternalKubeClientSet(client kubernetes.Interface) { p.client = client }

// InspectFeatureGates configures the plugin for the checkpoint feature.
func (p *Plugin) InspectFeatureGates(gates featuregate.FeatureGate) {
	p.enabled = gates.Enabled(features.PodLevelCheckpointRestore)
	p.inspectedFeatureGates = true
}

// ValidateInitialization checks that admission dependencies were supplied.
func (p *Plugin) ValidateInitialization() error {
	if p.client == nil {
		return fmt.Errorf("%s requires a Kubernetes client", PluginName)
	}
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s requires feature gates", PluginName)
	}
	return nil
}

// Validate rejects requests that cannot yet be processed by a source kubelet
// and Pod spec changes during capture.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.enabled {
		return nil
	}
	if a.GetOperation() == admission.Update && a.GetResource().GroupResource() == api.Resource("pods") {
		return validateCheckpointingUpdate(o.GetObjectConvertor(), a)
	}
	if a.GetOperation() != admission.Create || a.GetSubresource() != "" || a.GetResource().GroupResource() != node.Resource("podcheckpoints") {
		return nil
	}
	checkpoint, ok := a.GetObject().(*node.PodCheckpoint)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected PodCheckpoint, got %T", a.GetObject()))
	}
	if checkpoint.Spec.SourcePod == nil || checkpoint.Spec.SourcePod.Name == "" {
		return admission.NewForbidden(a, fmt.Errorf("spec.sourcePod.name is required"))
	}
	ref := checkpoint.Spec.SourcePod
	// Read live status so a request immediately following Pod startup need not
	// wait for a separate admission informer to catch up.
	pod, err := p.client.CoreV1().Pods(a.GetNamespace()).Get(ctx, ref.Name, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("cannot read source Pod %q: %w", ref.Name, err))
	}
	if ref.UID != nil && *ref.UID != pod.UID {
		return admission.NewForbidden(a, fmt.Errorf("source Pod %q has UID %q, not requested UID %q", ref.Name, pod.UID, *ref.UID))
	}
	if pod.DeletionTimestamp != nil {
		return admission.NewForbidden(a, fmt.Errorf("source Pod %q is being deleted", ref.Name))
	}
	if pod.Spec.NodeName == "" || pod.Status.Phase != v1.PodRunning {
		return admission.NewForbidden(a, fmt.Errorf("source Pod %q must be assigned to a node and Running before checkpointing", ref.Name))
	}
	if len(checkpoint.Spec.CheckpointOptions) == 0 {
		return nil
	}
	if pod.Spec.RuntimeClassName == nil || *pod.Spec.RuntimeClassName == "" {
		return admission.NewForbidden(a, fmt.Errorf("spec.checkpointOptions requires a source Pod with spec.runtimeClassName and a RuntimeClass checkpoint option allowlist"))
	}
	className := *pod.Spec.RuntimeClassName
	class, err := p.client.NodeV1().RuntimeClasses().Get(ctx, className, metav1.GetOptions{})
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("cannot read RuntimeClass %q for spec.checkpointOptions: %w", className, err))
	}
	var allowed []string
	if class.PodCheckpointOptions != nil {
		allowed = class.PodCheckpointOptions.AllowedCheckpointOptions
	}
	if err := checkpointutil.ValidateRuntimeOptions(checkpoint.Spec.CheckpointOptions, allowed); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("spec.checkpointOptions for RuntimeClass %q: %w", className, err))
	}
	return nil
}

// validateCheckpointingUpdate preserves the workload while the kubelet captures
// it. The old status determines whether capture is active, so an update cannot
// clear the condition to bypass the check. Status and binding updates have their
// own authorization and validation.
func validateCheckpointingUpdate(convertor runtime.ObjectConvertor, a admission.Attributes) error {
	switch a.GetSubresource() {
	case "", "resize", "ephemeralcontainers":
	default:
		return nil
	}
	pod, ok := a.GetObject().(*api.Pod)
	oldPod, oldOK := a.GetOldObject().(*api.Pod)
	if !ok || !oldOK {
		return nil
	}
	var checkpointing *api.PodCondition
	for i := range oldPod.Status.Conditions {
		condition := &oldPod.Status.Conditions[i]
		if condition.Type == api.PodCheckpointing && condition.Status == api.ConditionTrue {
			checkpointing = condition
			break
		}
	}
	if checkpointing == nil {
		return nil
	}

	var want, got v1.PodSpec
	if err := convertor.Convert(&oldPod.Spec, &want, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert existing Pod spec: %w", err))
	}
	if err := convertor.Convert(&pod.Spec, &got, nil); err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to convert updated Pod spec: %w", err))
	}
	// Gates delay scheduling and are not part of the captured workload.
	want.SchedulingGates, got.SchedulingGates = nil, nil
	if changedFields := checkpointutil.PodSpecChangedFields(&want, &got); len(changedFields) != 0 {
		message := "pod spec cannot change while Checkpointing=True"
		if checkpointing.Message != "" {
			message += ": " + checkpointing.Message
		}
		return admission.NewForbidden(a, fmt.Errorf("%s; scheduling gates may be removed; changed fields: %s", message, strings.Join(changedFields, ", ")))
	}
	return nil
}
