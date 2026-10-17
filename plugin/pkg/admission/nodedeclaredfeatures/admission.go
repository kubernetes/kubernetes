/*
Copyright 2025 The Kubernetes Authors.

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

package nodedeclaredfeatures

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/version"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/dranodeallocatableresources"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	apisresource "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PluginName is the name of this admission controller plugin.
	PluginName = "NodeDeclaredFeatureValidator"
)

// Register registers a plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin()
	})
}

// Plugin is an admission controller that validates pod updates against node capabilities.
type Plugin struct {
	*admission.Handler
	nodeLister                             corev1listers.NodeLister
	nodeDeclaredFeatureFramework           *ndf.Framework
	nodeDeclaredFeatureGateEnabled         bool
	draNodeAllocatableResourcesGateEnabled bool
	version                                *versionutil.Version
}

var _ admission.ValidationInterface = &Plugin{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&Plugin{})
var _ genericadmissioninitializer.WantsFeatures = &Plugin{}

// New creates a new Plugin admission plugin
func NewPlugin() (*Plugin, error) {
	ver, err := versionutil.Parse(version.Get().String())
	if err != nil {
		return nil, fmt.Errorf("failed to parse version: %w", err)
	}
	return &Plugin{
		Handler:                      admission.NewHandler(admission.Create, admission.Update),
		nodeDeclaredFeatureFramework: ndf.DefaultFramework,
		version:                      ver,
	}, nil
}

// SetExternalKubeInformerFactory sets the informer factory for the plugin.
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	nodeInformer := f.Core().V1().Nodes()
	p.nodeLister = f.Core().V1().Nodes().Lister()
	p.SetReadyFunc(nodeInformer.Informer().HasSynced)
}

// SetFeatures sets the feature gates for the plugin.
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.nodeDeclaredFeatureGateEnabled = featureGates.Enabled(features.NodeDeclaredFeatures)
	p.draNodeAllocatableResourcesGateEnabled = featureGates.Enabled(features.DRANodeAllocatableResources)
}

// ValidateInitialization ensures that the plugin is properly initialized.
func (p *Plugin) ValidateInitialization() error {
	if p.nodeLister == nil {
		return fmt.Errorf("missing node lister for %s", PluginName)
	}

	if p.nodeDeclaredFeatureFramework == nil {
		p.nodeDeclaredFeatureFramework = ndf.DefaultFramework
	}
	return nil
}

// Validate is the core of the admission controller logic.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	// Ignore if the feature gate is not enabled.
	if !p.nodeDeclaredFeatureGateEnabled {
		return nil
	}

	groupRes := a.GetResource().GroupResource()

	if groupRes == core.Resource("pods") {
		return p.validatePod(a)
	}

	if groupRes == apisresource.Resource("resourceslices") {
		return p.validateResourceSlice(a)
	}

	return nil
}

func (p *Plugin) validatePod(a admission.Attributes) error {
	if a.GetOperation() != admission.Update {
		return nil
	}

	// Only validate updates to the main pod spec (subresource == "")
	// or the custom "resize" subresource.
	subresource := a.GetSubresource()
	if subresource != "" && subresource != "resize" {
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

	if oldPod.Generation == pod.Generation {
		// since generation is only incremented when the spec changes, we can skip validation if it doesnt.
		return nil
	}

	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	return p.validatePodUpdate(pod, oldPod, a)
}

func (p *Plugin) validatePodUpdate(pod, oldPod *core.Pod, a admission.Attributes) error {
	// Convert internal to external pods for the helper library.
	podV1 := &corev1.Pod{}
	if err := v1.Convert_core_Pod_To_v1_Pod(pod, podV1, nil); err != nil {
		return errors.NewBadRequest(fmt.Sprintf("failed to convert pod: %v", err))
	}
	oldPodV1 := &corev1.Pod{}
	if err := v1.Convert_core_Pod_To_v1_Pod(oldPod, oldPodV1, nil); err != nil {
		return errors.NewBadRequest(fmt.Sprintf("failed to convert oldPod: %v", err))
	}
	oldPodInfo := &ndf.PodInfo{Spec: &oldPodV1.Spec, Status: &oldPodV1.Status}
	newPodInfo := &ndf.PodInfo{Spec: &podV1.Spec, Status: &podV1.Status}
	reqs, err := p.nodeDeclaredFeatureFramework.InferForPodUpdate(oldPodInfo, newPodInfo, p.version)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to infer pod capability requirements: %w", err))
	}
	// If there are no specific feature requirements for this update, we're done.
	if reqs.IsEmpty() {
		return nil
	}
	node, err := p.nodeLister.Get(pod.Spec.NodeName)
	if err != nil {
		if errors.IsNotFound(err) {
			return admission.NewForbidden(a, fmt.Errorf("node %q not found", pod.Spec.NodeName))
		}
		return admission.NewForbidden(a, fmt.Errorf("failed to get node %q: %w", pod.Spec.NodeName, err))
	}
	result, err := p.nodeDeclaredFeatureFramework.MatchNode(reqs, node)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to match pod requirements against node %q: %w", node.Name, err))
	}
	if !result.IsMatch {
		return admission.NewForbidden(a, fmt.Errorf("pod update requires features %s which are not available on node %q", strings.Join(result.UnsatisfiedRequirements, ", "), node.Name))
	}

	return nil
}

func (p *Plugin) validateResourceSlice(a admission.Attributes) error {
	if a.GetOperation() != admission.Create && a.GetOperation() != admission.Update {
		return nil
	}

	slice, ok := a.GetObject().(*apisresource.ResourceSlice)
	if !ok {
		return errors.NewBadRequest(fmt.Sprintf("expected a ResourceSlice but got a %T", a.GetObject()))
	}

	return p.validateNodeAllocatableDRA(slice, a)
}

func (p *Plugin) validateNodeAllocatableDRA(slice *apisresource.ResourceSlice, a admission.Attributes) error {
	// cluster with the feature disabled.
	if !p.draNodeAllocatableResourcesGateEnabled {
		return nil
	}

	var nodes []*corev1.Node
	var nodesErr error
	// listAllNodes  queries and caches the node list once per admission check
	// to avoid redundant scans when multiple devices in the slice
	// require selector or multi-node matching. Called only if any of the
	// device has selector or allNodes enabled.
	listAllNodes := func() ([]*corev1.Node, error) {
		if nodes != nil || nodesErr != nil {
			return nodes, nodesErr
		}
		nodes, nodesErr = p.nodeLister.List(labels.Everything())
		return nodes, nodesErr
	}

	for _, dev := range slice.Spec.Devices {
		if len(dev.NodeAllocatableResources) == 0 {
			continue
		}

		if !p.WaitForReady() {
			return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
		}

		var singleNode string
		var nodeSelector *core.NodeSelector
		var allNodes bool

		if slice.Spec.PerDeviceNodeSelection != nil && *slice.Spec.PerDeviceNodeSelection {
			if dev.NodeName != nil && *dev.NodeName != "" {
				singleNode = *dev.NodeName
			} else if dev.NodeSelector != nil {
				nodeSelector = dev.NodeSelector
			} else if dev.AllNodes != nil && *dev.AllNodes {
				allNodes = true
			}
		} else {
			if slice.Spec.NodeName != nil && *slice.Spec.NodeName != "" {
				singleNode = *slice.Spec.NodeName
			} else if slice.Spec.NodeSelector != nil {
				nodeSelector = slice.Spec.NodeSelector
			} else if slice.Spec.AllNodes != nil && *slice.Spec.AllNodes {
				allNodes = true
			}
		}

		if singleNode == "" && nodeSelector == nil && !allNodes {
			return admission.NewForbidden(a, fmt.Errorf("device %q in ResourceSlice has NodeAllocatableResources but does not map to any nodes", dev.Name))
		}

		if allNodes || (nodeSelector != nil) {
			nodes, err := listAllNodes()
			if err != nil {
				return admission.NewForbidden(a, fmt.Errorf("failed to list nodes for device %q: %w", dev.Name, err))
			}
			if err := p.validateNodeAllocatableDRAMultiNodeDevice(nodes, nodeSelector, allNodes, a); err != nil {
				return err
			}
		} else {
			node, err := p.nodeLister.Get(singleNode)
			if err != nil {
				if errors.IsNotFound(err) {
					return admission.NewForbidden(a, fmt.Errorf("node %q not found for device %q", singleNode, dev.Name))
				}
				return admission.NewForbidden(a, fmt.Errorf("failed to get node %q for device %q: %w", singleNode, dev.Name, err))
			}
			if err := p.validateNodeAllocatableDRADeclaredFeature(node, a); err != nil {
				return err
			}
		}
	}

	return nil
}

func (p *Plugin) validateNodeAllocatableDRADeclaredFeature(node *corev1.Node, a admission.Attributes) error {
	if !slices.Contains(node.Status.DeclaredFeatures, dranodeallocatableresources.DRANodeAllocatableResourcesFeature) {
		return admission.NewForbidden(a, fmt.Errorf("node %q does not support %s feature", node.Name, dranodeallocatableresources.DRANodeAllocatableResourcesFeature))
	}
	return nil
}

func (p *Plugin) validateNodeAllocatableDRAMultiNodeDevice(nodes []*corev1.Node, nodeSelector *core.NodeSelector, allNodes bool, a admission.Attributes) error {
	var selector *nodeaffinity.NodeSelector
	if nodeSelector != nil {
		v1Selector := &corev1.NodeSelector{}
		if err := v1.Convert_core_NodeSelector_To_v1_NodeSelector(nodeSelector, v1Selector, nil); err != nil {
			return errors.NewBadRequest(fmt.Sprintf("failed to convert node selector: %v", err))
		}
		var err error
		selector, err = nodeaffinity.NewNodeSelector(v1Selector)
		if err != nil {
			return errors.NewBadRequest(fmt.Sprintf("failed to create node selector: %v", err))
		}
	}
	for _, node := range nodes {
		match := allNodes
		if !match && selector != nil {
			match = selector.Match(node)
		}
		if match {
			if err := p.validateNodeAllocatableDRADeclaredFeature(node, a); err != nil {
				return err
			}
		}
	}
	return nil
}
