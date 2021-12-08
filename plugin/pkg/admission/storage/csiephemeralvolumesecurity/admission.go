/*
Copyright 2021 The Kubernetes Authors.

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

package podsecurity

import (
	"context"
	"fmt"
	"io"
	"sync"

	storagev1listers "k8s.io/client-go/listers/storage/v1"

	// install conversions for types we need to convert
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/component-base/metrics/legacyregistry"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	podsecurityadmission "k8s.io/pod-security-admission/admission"
	podsecurityconfigloader "k8s.io/pod-security-admission/admission/api/load"
	"k8s.io/pod-security-admission/api"
	podsecurityadmissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/metrics"
	"k8s.io/pod-security-admission/policy"
)

// PluginName is a string with the name of the plugin
const PluginName = "CSIEphemeralVolumeSecurity"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(reader io.Reader) (admission.Interface, error) {
		return newPlugin(reader)
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler

	enabled               bool
	inspectedFeatureGates bool
	defaultPolicy         api.Policy

	client          kubernetes.Interface
	namespaceLister corev1listers.NamespaceLister
	csiDriverLister storagev1listers.CSIDriverLister

	namespaceGetter podsecurityadmission.NamespaceGetter
}

var _ admission.ValidationInterface = &Plugin{}
var _ genericadmissioninit.WantsExternalKubeInformerFactory = &Plugin{}
var _ genericadmissioninit.WantsExternalKubeClientSet = &Plugin{}

var (
	defaultRecorder     *metrics.PrometheusRecorder
	defaultRecorderInit sync.Once
)

func getDefaultRecorder() metrics.Recorder {
	// initialize and register to legacy metrics once
	defaultRecorderInit.Do(func() {
		defaultRecorder = metrics.NewPrometheusRecorder(podsecurityadmissionapi.GetAPIVersion())
		defaultRecorder.MustRegister(legacyregistry.MustRegister)
	})
	return defaultRecorder
}

// newPlugin creates a new admission plugin.
func newPlugin(reader io.Reader) (*Plugin, error) {
	// this configuration will need be to essentially the same format
	config, err := podsecurityconfigloader.LoadFromReader(reader)
	if err != nil {
		return nil, err
	}

	evaluator, err := policy.NewEvaluator(policy.DefaultChecks())
	if err != nil {
		return nil, fmt.Errorf("could not create PodSecurityRegistry: %w", err)
	}

	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}, nil
}

// SetExternalKubeInformerFactory registers an informer
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	p.namespaceLister = namespaceInformer.Lister()
	csiDriverInformer := f.Storage().V1().CSIDrivers()
	p.csiDriverLister = csiDriverInformer.Lister()
	p.SetReadyFunc(func() bool {
		return namespaceInformer.Informer().HasSynced() && csiDriverInformer.Informer().HasSynced()
	})
}

// SetExternalKubeClientSet sets the plugin's client
func (p *Plugin) SetExternalKubeClientSet(client kubernetes.Interface) {
	p.client = client
}

func (c *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	c.enabled = featureGates.Enabled(features.PodSecurity)
	c.inspectedFeatureGates = true
}

// ValidateInitialization ensures all required options are set
func (p *Plugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s did not see feature gates", PluginName)
	}
	// return early if we don't have what we need to set up the admission delegate
	if p.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if p.csiDriverLister == nil {
		return fmt.Errorf("missing csiDriverLister")
	}
	if p.client == nil {
		return fmt.Errorf("missing client")
	}

	p.namespaceGetter = podsecurityadmission.NamespaceGetterFromListerAndClient(p.namespaceLister, p.client)

	return nil
}

var (
	applicableResources = map[schema.GroupResource]bool{
		corev1.Resource("pods"): true,
	}
)

func (p *Plugin) PolicyToEvaluate(labels map[string]string) (api.Policy, field.ErrorList) {
	return api.PolicyToEvaluate(labels, p.defaultPolicy)
}

func (p *Plugin) Validate(ctx context.Context, attrs admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.enabled {
		return nil
	}
	gr := attrs.GetResource().GroupResource()
	if !applicableResources[gr] {
		return nil
	}

	pod, ok := attrs.GetObject().(*coreapi.Pod)
	if !ok {
		return admission.NewForbidden(attrs, fmt.Errorf("not a pod: %T", attrs.GetObject()))
	}

	namespace, err := p.namespaceGetter.GetNamespace(ctx, pod.Namespace)
	if err != nil {
		return admission.NewForbidden(attrs, err)
	}

	nsPolicy, nsPolicyErrs := p.PolicyToEvaluate(namespace.Labels)

	// for each container
	//     for each ephemeral volume
	//         find the owning csidriver using the csiLister
	//         check to be sure it matches the level allowed by the namespace, create errors, warnings, and audit

	return nil
}
