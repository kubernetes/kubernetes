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

package podcertificatemaxexpiration

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/component-base/featuregate"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

// PluginName names the plugin
const PluginName = "PodCertificateMaxExpiration"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

type Plugin struct {
	*admission.Handler
	inspectedFeatureGates bool
	enabled               bool
}

var _ admission.MutationInterface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ admission.InitializationValidator = &Plugin{}
var _ genericadmissioninit.WantsFeatures = &Plugin{}

// InspectFeatureGates implements WantsFeatures.
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.enabled = featureGates.Enabled(features.PodCertificateRequest)
	p.inspectedFeatureGates = true
}

// ValidateInitialization checks that the plugin has been correctly initialized.
func (p *Plugin) ValidateInitialization() error {
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s did not see feature gates", PluginName)
	}
	return nil
}

// NewPlugin creates a new CSR approval admission plugin
func NewPlugin() *Plugin {
	return &Plugin{}
}

var pcrGroupResource = api.Resource("podcertificaterequests")

// Admit mutates incoming PodCertificateRequests to impose a maximum lifetime if
// one is configured.
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetResource().GroupResource() != pcrGroupResource {
		return nil
	}
	if a.GetOperation() != admission.Create {
		return nil
	}

	// PodCertificateRequests are immutable after create, so we only need to
	// worry about creation requests.

	pcr, ok := a.GetObject().(*api.PodCertificateRequest)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected type PodCertificateRequest, got %T", a.GetObject()))
	}

	// TODO(KEP-4317): Per-signer configurable max lifetimes
	pcr.Spec.MaxExpirationSeconds = ptr.To[int32](86400)

	return nil
}

// Validate checks incoming PodCertificateRequests to ensure that they adhere to
// the configured maximum lifetime for the signer.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if a.GetResource().GroupResource() != pcrGroupResource {
		return nil
	}
	if a.GetOperation() != admission.Create {
		return nil
	}

	// PodCertificateRequests are immutable after create, so we only need to
	// worry about creation requests.

	pcr, ok := a.GetObject().(*api.PodCertificateRequest)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected type PodCertificateRequest, got %T", a.GetObject()))
	}

	// TODO(KEP-4317): Per-signer configurable max lifetimes.
	if pcr.Spec.MaxExpirationSeconds == nil || *pcr.Spec.MaxExpirationSeconds > 86400 {
		return admission.NewForbidden(a, fmt.Errorf("PodCertificateRequest for signer %s must have spec.MaxExpirationSeconds <= %d", pcr.Spec.SignerName, 86400))
	}

	return nil
}
