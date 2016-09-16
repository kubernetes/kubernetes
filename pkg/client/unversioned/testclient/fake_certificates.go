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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// NewSimpleFakeCertificate returns a client that will respond with the provided objects
func NewSimpleFakeCertificates(objects ...runtime.Object) *FakeCertificates {
	return &FakeCertificates{Fake: NewSimpleFake(objects...)}
}

// FakeCertificates implements CertificatesInterface. Meant to be
// embedded into a struct to get a default implementation. This makes faking
// out just the method you want to test easier.
type FakeCertificates struct {
	*Fake
}

func (c *FakeCertificates) CertificateSigningRequests() unversioned.CertificateSigningRequestInterface {
	return &FakeCertificateSigningRequest{Fake: c}
}

// FakeCertificateSigningRequest implements CertificateSigningRequestInterface
type FakeCertificateSigningRequest struct {
	Fake *FakeCertificates
}

func (c *FakeCertificateSigningRequest) Get(name string) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.Invokes(NewRootGetAction("certificatesigningrequests", name), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}

	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequest) List(opts api.ListOptions) (*certificates.CertificateSigningRequestList, error) {
	obj, err := c.Fake.Invokes(NewRootListAction("certificatesigningrequests", opts), &certificates.CertificateSigningRequestList{})
	if obj == nil {
		return nil, err
	}

	return obj.(*certificates.CertificateSigningRequestList), err
}

func (c *FakeCertificateSigningRequest) Create(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.Invokes(NewRootCreateAction("certificatesigningrequests", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequest) Update(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.Invokes(NewRootUpdateAction("certificatesigningrequests", csr), csr)
	if obj == nil {
		return nil, err
	}

	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequest) UpdateStatus(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("certificatesigningrequests", "status", api.NamespaceAll, csr), csr)
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequest) UpdateApproval(csr *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.Invokes(NewUpdateSubresourceAction("certificatesigningrequests", "approval", api.NamespaceAll, csr), csr)
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequest) Delete(name string, opts *api.DeleteOptions) error {
	_, err := c.Fake.Invokes(NewRootDeleteAction("certificatesigningrequests", name), &certificates.CertificateSigningRequest{})
	return err
}

func (c *FakeCertificateSigningRequest) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.InvokesWatch(NewRootWatchAction("certificatesigningrequests", opts))
}
