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

package unversioned

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/watch"
)

// CertificateSigningRequestInterface has methods to work with CertificateSigningRequest resources.
type CertificateSigningRequestInterface interface {
	List(opts api.ListOptions) (*certificates.CertificateSigningRequestList, error)
	Get(name string) (*certificates.CertificateSigningRequest, error)
	Delete(name string, options *api.DeleteOptions) error
	Create(certificateSigningRequest *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
	Update(certificateSigningRequest *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
	UpdateStatus(certificateSigningRequest *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
	UpdateApproval(certificateSigningRequest *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error)
	Watch(opts api.ListOptions) (watch.Interface, error)
}

// certificateSigningRequests implements CertificateSigningRequestsNamespacer interface
type certificateSigningRequests struct {
	client *CertificatesClient
}

// newCertificateSigningRequests returns a certificateSigningRequests
func newCertificateSigningRequests(c *CertificatesClient) *certificateSigningRequests {
	return &certificateSigningRequests{
		client: c,
	}
}

// List takes label and field selectors, and returns the list of certificateSigningRequests that match those selectors.
func (c *certificateSigningRequests) List(opts api.ListOptions) (result *certificates.CertificateSigningRequestList, err error) {
	result = &certificates.CertificateSigningRequestList{}
	err = c.client.Get().Resource("certificatesigningrequests").VersionedParams(&opts, api.ParameterCodec).Do().Into(result)
	return
}

// Get takes the name of the certificateSigningRequest, and returns the corresponding CertificateSigningRequest object, and an error if it occurs
func (c *certificateSigningRequests) Get(name string) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Get().Resource("certificatesigningrequests").Name(name).Do().Into(result)
	return
}

// Delete takes the name of the certificateSigningRequest and deletes it.  Returns an error if one occurs.
func (c *certificateSigningRequests) Delete(name string, options *api.DeleteOptions) error {
	return c.client.Delete().Resource("certificatesigningrequests").Name(name).Body(options).Do().Error()
}

// Create takes the representation of a certificateSigningRequest and creates it.  Returns the server's representation of the certificateSigningRequest, and an error, if it occurs.
func (c *certificateSigningRequests) Create(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Post().Resource("certificatesigningrequests").Body(certificateSigningRequest).Do().Into(result)
	return
}

// Update takes the representation of a certificateSigningRequest and updates it.  Returns the server's representation of the certificateSigningRequest, and an error, if it occurs.
func (c *certificateSigningRequests) Update(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Put().Resource("certificatesigningrequests").Name(certificateSigningRequest.Name).Body(certificateSigningRequest).Do().Into(result)
	return
}

// UpdateStatus takes the representation of a certificateSigningRequest and updates it.  Returns the server's representation of the certificateSigningRequest, and an error, if it occurs.
func (c *certificateSigningRequests) UpdateStatus(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Put().Resource("certificatesigningrequests").Name(certificateSigningRequest.Name).SubResource("status").Body(certificateSigningRequest).Do().Into(result)
	return
}

// UpdateApproval takes the representation of a certificateSigningRequest and updates it.  Returns the server's representation of the certificateSigningRequest, and an error, if it occurs.
func (c *certificateSigningRequests) UpdateApproval(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	result = &certificates.CertificateSigningRequest{}
	err = c.client.Put().Resource("certificatesigningrequests").Name(certificateSigningRequest.Name).SubResource("approval").Body(certificateSigningRequest).Do().Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested certificateSigningRequests.
func (c *certificateSigningRequests) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.client.Get().
		Prefix("watch").
		Namespace(api.NamespaceAll).
		Resource("certificatesigningrequests").
		VersionedParams(&opts, api.ParameterCodec).
		Watch()
}
