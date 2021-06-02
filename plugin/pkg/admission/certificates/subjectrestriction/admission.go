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

package subjectrestriction

import (
	"context"
	"fmt"
	"io"

	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/klog/v2"
	certificatesapi "k8s.io/kubernetes/pkg/apis/certificates"
)

// PluginName is a string with the name of the plugin
const PluginName = "CertificateSubjectRestriction"

// Register registers the plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(), nil
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
}

// ValidateInitialization always returns nil.
func (p *Plugin) ValidateInitialization() error {
	return nil
}

var _ admission.ValidationInterface = &Plugin{}

// NewPlugin constructs a new instance of the CertificateSubjectRestrictions admission interface.
func NewPlugin() *Plugin {
	return &Plugin{
		Handler: admission.NewHandler(admission.Create),
	}
}

var csrGroupResource = certificatesapi.Resource("certificatesigningrequests")

// Validate ensures that if the signerName on a CSR is set to
// `kubernetes.io/kube-apiserver-client`, that its organization (group)
// attribute is not set to `system:masters`.
func (p *Plugin) Validate(_ context.Context, a admission.Attributes, _ admission.ObjectInterfaces) error {
	if a.GetResource().GroupResource() != csrGroupResource || a.GetSubresource() != "" {
		return nil
	}

	csr, ok := a.GetObject().(*certificatesapi.CertificateSigningRequest)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("expected type CertificateSigningRequest, got: %T", a.GetObject()))
	}

	if csr.Spec.SignerName != certificatesv1beta1.KubeAPIServerClientSignerName {
		return nil
	}

	csrParsed, err := certificatesapi.ParseCSR(csr.Spec.Request)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("failed to parse CSR: %v", err))
	}

	for _, group := range csrParsed.Subject.Organization {
		if group == "system:masters" {
			klog.V(4).Infof("CSR %s rejected by admission plugin %s for attempting to use signer %s with system:masters group",
				csr.Name, PluginName, certificatesv1beta1.KubeAPIServerClientSignerName)
			return admission.NewForbidden(a, fmt.Errorf("use of %s signer with system:masters group is not allowed",
				certificatesv1beta1.KubeAPIServerClientSignerName))
		}
	}

	return nil
}
