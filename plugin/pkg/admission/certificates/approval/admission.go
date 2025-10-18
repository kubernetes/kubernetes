/*
Copyright 2020 The Kubernetes Authors.

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

package approval

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/certauthorization"
)

// PluginName is a string with the name of the plugin
const PluginName = "CertificateApproval"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
	authz authorizer.Authorizer
}

// SetAuthorizer sets the authorizer.
func (p *Plugin) SetAuthorizer(authz authorizer.Authorizer) {
	p.authz = authz
}

// ValidateInitialization ensures an authorizer is set.
func (p *Plugin) ValidateInitialization() error {
	if p.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	return nil
}

var _ admission.ValidationInterface = &Plugin{}
var _ genericadmissioninit.WantsAuthorizer = &Plugin{}

// NewPlugin creates a new CSR approval admission plugin
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Update),
	}
}

var csrGroupResource = api.Resource("certificatesigningrequests")

// Validate verifies that the requesting user has permission to approve
// CertificateSigningRequests for the specified signerName.
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	// Ignore all calls to anything other than 'certificatesigningrequests/approval'.
	// Ignore all operations other than UPDATE.
	if a.GetSubresource() != "approval" ||
		a.GetResource().GroupResource() != csrGroupResource {
		return nil
	}

	// We check permissions against the *old* version of the resource, in case
	// a user is attempting to update the SignerName when calling the approval
	// endpoint (which is an invalid/not allowed operation)
	csr, ok := a.GetOldObject().(*api.CertificateSigningRequest)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected type CertificateSigningRequest, got: %T", a.GetOldObject()))
	}

	if !certauthorization.IsAuthorizedForSignerName(ctx, p.authz, a.GetUserInfo(), "approve", csr.Spec.SignerName) {
		klog.V(4).Infof("user not permitted to approve CertificateSigningRequest %q with signerName %q", csr.Name, csr.Spec.SignerName)
		return admission.NewForbidden(a, fmt.Errorf("user not permitted to approve requests with signerName %q", csr.Spec.SignerName))
	}

	return nil
}
