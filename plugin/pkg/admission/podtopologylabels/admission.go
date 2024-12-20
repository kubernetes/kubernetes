/*
Copyright 2024 The Kubernetes Authors.

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

package podtopologylabels

import (
	"context"
	"fmt"
	"io"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// PluginName is a string with the name of the plugin
const PluginName = "PodTopologyLabels"

// defaultConfig is the configuration used for the default instantiation of the plugin.
// This configuration is used by kube-apiserver.
// It is not exported to avoid any chance of accidentally mutating the variable.
var defaultConfig = Config{
	Labels: []string{"topology.kubernetes.io/zone", "topology.kubernetes.io/region"},
}

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(_ io.Reader) (admission.Interface, error) {
		plugin := NewPodTopologyPlugin(defaultConfig)
		return plugin, nil
	})
}

// Config contains configuration for instances of the podtopologylabels admission plugin.
// This is not exposed as user-facing APIServer configuration, however can be used by
// platform operators when building custom topology label plugins.
type Config struct {
	// Labels is set of explicit label keys to be copied from the Node object onto
	// pod Binding objects during admission.
	Labels []string
}

// NewPodTopologyPlugin initializes a Plugin
func NewPodTopologyPlugin(c Config) *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create),
		labels:  sets.New(c.Labels...),
	}
}

type Plugin struct {
	*admission.Handler

	nodeLister corev1listers.NodeLister

	// explicit labels to be copied to Pod objects being bound.
	labels sets.Set[string]

	enabled, inspectedFeatureGates bool
}

var _ admission.MutationInterface = &Plugin{}
var _ genericadmissioninitializer.WantsExternalKubeInformerFactory = &Plugin{}
var _ genericadmissioninitializer.WantsFeatures = &Plugin{}

// InspectFeatureGates implements WantsFeatures.
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.enabled = featureGates.Enabled(features.PodTopologyLabelsAdmission)
	p.inspectedFeatureGates = true
}

func (p *Plugin) SetExternalKubeInformerFactory(factory informers.SharedInformerFactory) {
	nodeInformer := factory.Core().V1().Nodes()
	p.nodeLister = nodeInformer.Lister()
	p.SetReadyFunc(nodeInformer.Informer().HasSynced)
}

func (p *Plugin) ValidateInitialization() error {
	if p.nodeLister == nil {
		return fmt.Errorf("nodeLister not set")
	}
	if !p.inspectedFeatureGates {
		return fmt.Errorf("feature gates not inspected")
	}
	return nil
}

func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if !p.enabled {
		return nil
	}
	// check whether the request is for a Binding or a Pod spec update.
	shouldAdmit, doAdmit, err := p.shouldAdmit(a)
	if !shouldAdmit || err != nil {
		// error is either nil and admit == false, or err is non-nil and should be returned.
		return err
	}
	// we need to wait for our caches to warm
	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}
	// run type specific admission
	return doAdmit(ctx, a, o)
}

func (p *Plugin) admitBinding(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	binding := a.GetObject().(*api.Binding)
	// other fields are not set by the default scheduler for the binding target, so only check the Kind.
	if binding.Target.Kind != "Node" {
		klog.V(6).Info("Skipping Pod being bound to non-Node object type", "target", binding.Target.GroupVersionKind())
		return nil
	}

	labelsToCopy, err := p.topologyLabelsForNodeName(binding.Target.Name)
	if err != nil {
		return err
	}
	if len(labelsToCopy) == 0 {
		// fast-path/short circuit if the node has no topology labels
		return nil
	}

	// copy the topology labels into the Binding's labels, as these are copied from the Binding
	// to the Pod object being bound within the podBinding registry/store.
	if binding.Labels == nil {
		binding.Labels = make(map[string]string)
	}
	mergeLabels(binding.Labels, labelsToCopy)

	return nil
}

func (p *Plugin) admitPod(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	pod := a.GetObject().(*api.Pod)
	if pod.Spec.NodeName == "" {
		// pod has not been scheduled yet
		return nil
	}

	// Determine the topology labels from the assigned node to be copied.
	labelsToCopy, err := p.topologyLabelsForNodeName(pod.Spec.NodeName)
	if err != nil {
		return err
	}
	if len(labelsToCopy) == 0 {
		// fast-path/short circuit if the node has no topology labels
		return nil
	}

	// copy the topology labels into the Pod's labels.
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}
	// overwrite any existing labels on the pod
	mergeLabels(pod.Labels, labelsToCopy)

	return nil
}

func (p *Plugin) topologyLabelsForNodeName(nodeName string) (map[string]string, error) {
	labels := make(map[string]string)
	node, err := p.nodeLister.Get(nodeName)
	if err != nil {
		// Ignore NotFound errors to avoid risking breaking compatibility/behaviour.
		if apierrors.IsNotFound(err) {
			return labels, nil
		}
		return nil, err
	}

	for k, v := range node.Labels {
		if !p.isTopologyLabel(k) {
			continue
		}
		labels[k] = v
	}

	return labels, nil
}

// mergeLabels merges new into existing, overwriting existing keys.
func mergeLabels(existing, new map[string]string) {
	for k, v := range new {
		existing[k] = v
	}
}

func (p *Plugin) isTopologyLabel(key string) bool {
	// First check explicit label keys.
	if p.labels.Has(key) {
		return true
	}
	return false
}

type admitFunc func(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) (err error)

// shouldAdmit inspects the provided adminssion attributes to determine whether the request
// requires admittance through this plugin.
func (p *Plugin) shouldAdmit(a admission.Attributes) (bool, admitFunc, error) {
	if a.GetResource().GroupResource() != api.Resource("pods") {
		return false, nil, nil
	}

	switch a.GetSubresource() {
	case "": // regular Pod endpoint
		_, ok := a.GetObject().(*api.Pod)
		if !ok {
			return false, nil, fmt.Errorf("expected Pod but got %T", a.GetObject())
		}
		return true, p.admitPod, nil
	case "binding":
		_, ok := a.GetObject().(*api.Binding)
		if !ok {
			return false, nil, fmt.Errorf("expected Binding but got %s", a.GetKind().Kind)
		}
		return true, p.admitBinding, nil
	default:
		// Ignore all other sub-resources.
		return false, nil, nil
	}
}
