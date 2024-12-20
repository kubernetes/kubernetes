/*
Copyright 2022 The Kubernetes Authors.

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

package ctbattest

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/certauthorization"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/rbac"
)

const PluginName = "ClusterTrustBundleAttest"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// Plugin is the ClusterTrustBundle attest plugin.
//
// In order to create or update a ClusterTrustBundle that sets signerName,
// you must have the following permission: group=certificates.k8s.io
// resource=signers resourceName=<the signer name> verb=attest.
type Plugin struct {
	*admission.Handler
	authz authorizer.Authorizer

	inspectedFeatureGates bool
	enabled               bool
}

var _ admission.ValidationInterface = &Plugin{}
var _ admission.InitializationValidator = &Plugin{}
var _ genericadmissioninit.WantsAuthorizer = &Plugin{}
var _ genericadmissioninit.WantsFeatures = &Plugin{}

func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// SetAuthorizer sets the plugin's authorizer.
func (p *Plugin) SetAuthorizer(authz authorizer.Authorizer) {
	p.authz = authz
}

// InspectFeatureGates implements WantsFeatures.
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.enabled = featureGates.Enabled(features.ClusterTrustBundle)
	p.inspectedFeatureGates = true
}

// ValidateInitialization checks that the plugin was initialized correctly.
func (p *Plugin) ValidateInitialization() error {
	if p.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if !p.inspectedFeatureGates {
		return fmt.Errorf("%s did not see feature gates", PluginName)
	}
	return nil
}

var clusterTrustBundleGroupResource = api.Resource("clustertrustbundles")

func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	if !p.enabled {
		return nil
	}
	if a.GetResource().GroupResource() != clusterTrustBundleGroupResource {
		return nil
	}

	newBundle, ok := a.GetObject().(*api.ClusterTrustBundle)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected type ClusterTrustBundle, got: %T", a.GetOldObject()))
	}

	// Unlike CSRs, it's OK to validate against the *new* object, because
	// updates to signer name will be rejected during validation.

	// If signer name isn't specified, we don't need to perform the
	// attest check.
	if newBundle.Spec.SignerName == "" {
		return nil
	}

	// Skip the attest check when the semantics of the bundle are unchanged to support storage migration and GC workflows
	if a.GetOperation() == admission.Update && rbac.IsOnlyMutatingGCFields(a.GetObject(), a.GetOldObject(), kapihelper.Semantic) {
		return nil
	}

	if !certauthorization.IsAuthorizedForSignerName(ctx, p.authz, a.GetUserInfo(), "attest", newBundle.Spec.SignerName) {
		klog.V(4).Infof("user not permitted to attest ClusterTrustBundle %q with signerName %q", newBundle.Name, newBundle.Spec.SignerName)
		return admission.NewForbidden(a, fmt.Errorf("user not permitted to attest for signerName %q", newBundle.Spec.SignerName))
	}

	return nil
}
