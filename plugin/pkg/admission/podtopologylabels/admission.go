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
	"strings"

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
// This is the configured used by kube-apiserver.
// It is not exported to avoid any chance of accidentally mutating the variable.
var defaultConfig = Config{
	Labels: []string{"topology.k8s.io/zone", "topology.k8s.io/region", "kubernetes.io/hostname"},
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
	// Domains is a set of label key prefixes used to copy across label values
	// for all labels with a given domain prefix.
	// For example, `example.com` would match all labels matching `example.com/*`.
	// Keys without a domain portion (i.e. those not containing a /) will never match.
	Domains []string
	// Suffixes is a set of label key domain suffixes used to copy label values for
	// all labels of a given subdomain.
	// This acts as a suffix match on the domain portion of label keys.
	// If a suffix does not have a leading '.', one will be prepended to avoid
	// programmer errors with values like `example.com` matching `notexample.com`.
	// Keys without a domain portion (i.e. those not containing a /) will never match.
	Suffixes []string
}

// NewPodTopologyPlugin initializes a Plugin
func NewPodTopologyPlugin(c Config) *Plugin {
	return &Plugin{
		Handler:  admission.NewHandler(admission.Create),
		labels:   sets.New(c.Labels...),
		domains:  sets.New(c.Domains...),
		suffixes: sets.New(c.Suffixes...),
	}
}

type Plugin struct {
	*admission.Handler

	nodeLister corev1listers.NodeLister

	// explicit labels, list of domains or a list of domain
	// suffixes to be copies to Pod objects being bound.
	labels, domains, suffixes sets.Set[string]

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
	if shouldIgnore(a) {
		return nil
	}
	// we need to wait for our caches to warm
	if !p.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	binding := a.GetObject().(*api.Binding)
	// other fields are not set by the default scheduler for the binding target, so only check the Kind.
	if binding.Target.Kind != "Node" {
		klog.V(6).Info("Skipping Pod being bound to non-Node object type", "target", binding.Target.GroupVersionKind())
		return nil
	}

	node, err := p.nodeLister.Get(binding.Target.Name)
	if err != nil {
		// Ignore NotFound errors to avoid risking breaking compatibility/behaviour.
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	// fast-path/short circuit if the node has no labels
	if node.Labels == nil {
		return nil
	}

	labelsToCopy := make(map[string]string)
	for k, v := range node.Labels {
		if !p.isTopologyLabel(k) {
			continue
		}
		labelsToCopy[k] = v
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
	for k, v := range labelsToCopy {
		if _, exists := binding.Labels[k]; exists {
			// Don't overwrite labels on Binding resources as this could lead to unexpected
			// behaviour if any schedulers rely on being able to explicitly set values themselves.
			continue
		}
		binding.Labels[k] = v
	}

	return nil
}

func (p *Plugin) isTopologyLabel(key string) bool {
	// First check explicit label keys.
	if p.labels.Has(key) {
		return true
	}
	// Check the domain portion of the label key, if present
	domain, _, hasDomain := strings.Cut(key, "/")
	if !hasDomain {
		// fast-path if there is no / separator
		return false
	}
	if p.domains.Has(domain) {
		// check for explicit domains to copy
		return true
	}
	for _, suffix := range p.suffixes.UnsortedList() {
		// check if the domain has one of the suffixes that are to be copied
		if strings.HasSuffix(domain, suffix) {
			return true
		}
	}
	return false
}

func shouldIgnore(a admission.Attributes) bool {
	resource := a.GetResource().GroupResource()
	if resource != api.Resource("pods") {
		return true
	}
	if a.GetSubresource() != "binding" {
		// only run the checks below on the binding subresource
		return true
	}

	obj := a.GetObject()
	_, ok := obj.(*api.Binding)
	if !ok {
		klog.Errorf("expected Binding but got %s", a.GetKind().Kind)
		return true
	}

	return false
}
