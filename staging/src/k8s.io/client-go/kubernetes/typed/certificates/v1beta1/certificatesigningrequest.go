/*
Copyright 2017 The Kubernetes Authors.

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

package v1beta1

import (
	v1beta1 "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	scheme "k8s.io/client-go/kubernetes/scheme"
	rest "k8s.io/client-go/rest"
)

// CertificateSigningRequestsGetter has a method to return a CertificateSigningRequestInterface.
// A group's client should implement this interface.
type CertificateSigningRequestsGetter interface {
	CertificateSigningRequests() CertificateSigningRequestInterface
}

// CertificateSigningRequestInterface has methods to work with CertificateSigningRequest resources.
type CertificateSigningRequestInterface interface {
	Create(*v1beta1.CertificateSigningRequest) (*v1beta1.CertificateSigningRequest, error)
	Update(*v1beta1.CertificateSigningRequest) (*v1beta1.CertificateSigningRequest, error)
	UpdateStatus(*v1beta1.CertificateSigningRequest) (*v1beta1.CertificateSigningRequest, error)
	Delete(name string, options *v1.DeleteOptions) error
	DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error
	Get(name string, options v1.GetOptions) (*v1beta1.CertificateSigningRequest, error)
	List(opts v1.ListOptions) (*v1beta1.CertificateSigningRequestList, error)
	Watch(opts v1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.CertificateSigningRequest, err error)
	CertificateSigningRequestExpansion
}

// certificateSigningRequests implements CertificateSigningRequestInterface
type certificateSigningRequests struct {
	client rest.Interface
}

// newCertificateSigningRequests returns a CertificateSigningRequests
func newCertificateSigningRequests(c *CertificatesV1beta1Client) *certificateSigningRequests {
	return &certificateSigningRequests{
		client: c.RESTClient(),
	}
}

// Get takes name of the certificateSigningRequest, and returns the corresponding certificateSigningRequest object, and an error if there is any.
func (c *certificateSigningRequests) Get(name string, options v1.GetOptions) (result *v1beta1.CertificateSigningRequest, err error) {
	result = &v1beta1.CertificateSigningRequest{}
	err = c.client.Get().
		Resource("certificatesigningrequests").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of CertificateSigningRequests that match those selectors.
func (c *certificateSigningRequests) List(opts v1.ListOptions) (result *v1beta1.CertificateSigningRequestList, err error) {
	result = &v1beta1.CertificateSigningRequestList{}
	err = c.client.Get().
		Resource("certificatesigningrequests").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do().
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested certificateSigningRequests.
func (c *certificateSigningRequests) Watch(opts v1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.client.Get().
		Resource("certificatesigningrequests").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch()
}

// Create takes the representation of a certificateSigningRequest and creates it.  Returns the server's representation of the certificateSigningRequest, and an error, if there is any.
func (c *certificateSigningRequests) Create(certificateSigningRequest *v1beta1.CertificateSigningRequest) (result *v1beta1.CertificateSigningRequest, err error) {
	result = &v1beta1.CertificateSigningRequest{}
	err = c.client.Post().
		Resource("certificatesigningrequests").
		Body(certificateSigningRequest).
		Do().
		Into(result)
	return
}

// Update takes the representation of a certificateSigningRequest and updates it. Returns the server's representation of the certificateSigningRequest, and an error, if there is any.
func (c *certificateSigningRequests) Update(certificateSigningRequest *v1beta1.CertificateSigningRequest) (result *v1beta1.CertificateSigningRequest, err error) {
	result = &v1beta1.CertificateSigningRequest{}
	err = c.client.Put().
		Resource("certificatesigningrequests").
		Name(certificateSigningRequest.Name).
		Body(certificateSigningRequest).
		Do().
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().

func (c *certificateSigningRequests) UpdateStatus(certificateSigningRequest *v1beta1.CertificateSigningRequest) (result *v1beta1.CertificateSigningRequest, err error) {
	result = &v1beta1.CertificateSigningRequest{}
	err = c.client.Put().
		Resource("certificatesigningrequests").
		Name(certificateSigningRequest.Name).
		SubResource("status").
		Body(certificateSigningRequest).
		Do().
		Into(result)
	return
}

// Delete takes name of the certificateSigningRequest and deletes it. Returns an error if one occurs.
func (c *certificateSigningRequests) Delete(name string, options *v1.DeleteOptions) error {
	return c.client.Delete().
		Resource("certificatesigningrequests").
		Name(name).
		Body(options).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *certificateSigningRequests) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return c.client.Delete().
		Resource("certificatesigningrequests").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Body(options).
		Do().
		Error()
}

// Patch applies the patch and returns the patched certificateSigningRequest.
func (c *certificateSigningRequests) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1beta1.CertificateSigningRequest, err error) {
	result = &v1beta1.CertificateSigningRequest{}
	err = c.client.Patch(pt).
		Resource("certificatesigningrequests").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return
}
