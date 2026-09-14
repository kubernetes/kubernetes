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

package podresize

import (
	"context"
	stderrors "errors"
	"fmt"
	"io"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PluginName is the name of this admission controller plugin.
	PluginName = "PodResizeValidator"

	ReasonNodeCapacity        metav1.CauseType = "NodeCapacity"
	ReasonUnsupportedPlatform metav1.CauseType = "UnsupportedPlatform"
)

// Register registers a plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin()
	})
}

// Plugin is an admission controller that validates pod resize requests against the node's capabilities.
type Plugin struct {
	*admission.Handler
	nodeLister                       corev1listers.NodeLister
	inPlacePodVerticalScalingEnabled bool
}

var _ admission.ValidationInterface = &Plugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&Plugin{})

// New creates a new ResizeValidator admission plugin
func NewPlugin() (*Plugin, error) {
	return &Plugin{
		Handler: admission.NewHandler(admission.Update),
	}, nil
}

// SetExternalKubeInformerFactory sets the informer factory for the plugin.
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	nodeInformer := f.Core().V1().Nodes()
	p.nodeLister = nodeInformer.Lister()
	p.SetReadyFunc(nodeInformer.Informer().HasSynced)
}

// InspectFeatureGates sets the feature gates for the plugin.
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.inPlacePodVerticalScalingEnabled = featureGates.Enabled(features.InPlacePodVerticalScaling)
}

// ValidateInitialization ensures that the plugin is properly initialized.
func (p *Plugin) ValidateInitialization() error {
	if p.nodeLister == nil {
		return fmt.Errorf("missing node lister for %s", PluginName)
	}

	return nil
}

// Validate is the core of the admission controller logic.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	// Ignore if the feature gate is not enabled.
	if !p.inPlacePodVerticalScalingEnabled {
		return nil
	}

	if a.GetOperation() != admission.Update {
		return nil
	}

	// We only care about Pod updates.
	if a.GetResource().GroupResource() != core.Resource("pods") {
		return nil
	}

	// Only validate updates to the custom "resize" subresource.
	if a.GetSubresource() != "resize" {
		return nil
	}
	pod, ok := a.GetObject().(*core.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected a pod but got a %T", a.GetObject()))
	}

	// We only care about pods that are already bound to a node.
	if len(pod.Spec.NodeName) == 0 {
		return nil
	}

	oldPod, ok := a.GetOldObject().(*core.Pod)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected an old pod but got a %T", a.GetOldObject()))
	}

	// Since generation is only incremented when the spec changes, we can skip validation if it doesnt.
	if oldPod.Generation == pod.Generation {
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	return p.validatePodResize(pod, a)
}

func (p *Plugin) validatePodResize(pod *core.Pod, a admission.Attributes) error {
	node, err := p.nodeLister.Get(pod.Spec.NodeName)
	if err != nil {
		if errors.IsNotFound(err) {
			return admission.NewForbidden(a, fmt.Errorf("node %q not found", pod.Spec.NodeName))
		}
		return admission.NewForbidden(a, fmt.Errorf("failed to get node %q: %w", pod.Spec.NodeName, err))
	}

	// If the node is not a linux node, reject the update.
	if err := validateLinuxNode(node); err != nil {
		return resizeValidationError(a, err, ReasonUnsupportedPlatform)
	}

	// If the new requests are larger than the node allocatable, reject the update.
	if err := validateNodeAllocatable(pod, node); err != nil {
		return resizeValidationError(a, err, ReasonNodeCapacity)
	}

	return nil
}

// Returns nil if the pod's resource requests fit within the node's allocatable, err otherwise.
func validateNodeAllocatable(pod *core.Pod, node *corev1.Node) error {
	// Convert internal to external pods for the helper library.
	podV1 := &corev1.Pod{}
	if err := v1.Convert_core_Pod_To_v1_Pod(pod, podV1, nil); err != nil {
		return errors.NewBadRequest(fmt.Sprintf("failed to convert pod: %v", err))
	}
	return validateRequestsWithinAllocatable(podV1, node.Status.Allocatable)
}

// Returns nil if the node is a linux node, err otherwise.
func validateLinuxNode(node *corev1.Node) error {
	// This label should always be populated.
	val, ok := node.Labels[corev1.LabelOSStable]
	if !ok || val == "linux" {
		return nil
	}
	return fmt.Errorf("pod resize is only supported on linux nodes, node %q is %q", node.Name, val)
}

func resizeValidationError(a admission.Attributes, err error, reason metav1.CauseType) error {
	forbiddenErr := admission.NewForbidden(a, err)
	var statusErr *errors.StatusError
	if stderrors.As(forbiddenErr, &statusErr) {
		statusErr.ErrStatus.Details.Causes = append(statusErr.ErrStatus.Details.Causes, metav1.StatusCause{
			Type: reason,
		})
		return statusErr
	}
	return forbiddenErr
}

// validateRequestsWithinAllocatable returns an error if the pod requests exceed the allocatable
// capacity.
func validateRequestsWithinAllocatable(pod *corev1.Pod, allocatable corev1.ResourceList) error {
	cpuAllocatable := allocatable.Cpu()
	memAllocatable := allocatable.Memory()

	requests := resource.PodRequests(pod, resource.PodResourcesOptions{})
	cpuRequests := requests[corev1.ResourceCPU]
	memRequests := requests[corev1.ResourceMemory]

	var msg []string
	if cpuRequests.Cmp(*cpuAllocatable) > 0 {
		msg = append(msg, fmt.Sprintf("cpu, requested: %d, allocatable: %d", cpuRequests.MilliValue(), cpuAllocatable.MilliValue()))
	}
	if memRequests.Cmp(*memAllocatable) > 0 {
		msg = append(msg, fmt.Sprintf("memory, requested: %d, allocatable: %d", memRequests.Value(), memAllocatable.Value()))
	}
	if len(msg) > 0 {
		return fmt.Errorf("node didn't have enough allocatable resources: %s", strings.Join(msg, "; "))
	}
	return nil
}
