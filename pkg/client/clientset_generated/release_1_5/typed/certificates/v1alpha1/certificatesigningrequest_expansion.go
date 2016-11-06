/*
Copyright 2016 The Kubernetes Authors.

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

package v1alpha1

import (
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
)

type CertificateSigningRequestExpansion interface {
	UpdateApproval(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error)
}

func (c *certificateSigningRequests) UpdateApproval(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Put().
		Resource("certificatesigningrequests").
		Name(certificateSigningRequest.Name).
		Body(certificateSigningRequest).
		SubResource("approval").
		Do().
		Into(result)
	return
}
