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

package fake

import (
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	watch "k8s.io/apimachinery/pkg/watch"
	api "k8s.io/kubernetes/pkg/api"
	certificates "k8s.io/kubernetes/pkg/apis/certificates"
	core "k8s.io/kubernetes/pkg/client/testing/core"
)

// FakeCertificateSigningRequests implements CertificateSigningRequestInterface
type FakeCertificateSigningRequests struct {
	Fake *FakeCertificates
}

var certificatesigningrequestsResource = schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "", Resource: "certificatesigningrequests"}

func (c *FakeCertificateSigningRequests) Create(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootCreateAction(certificatesigningrequestsResource, certificateSigningRequest), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) Update(certificateSigningRequest *certificates.CertificateSigningRequest) (result *certificates.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateAction(certificatesigningrequestsResource, certificateSigningRequest), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) UpdateStatus(certificateSigningRequest *certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	obj, err := c.Fake.
		Invokes(core.NewRootUpdateSubresourceAction(certificatesigningrequestsResource, "status", certificateSigningRequest), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) Delete(name string, options *api.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(core.NewRootDeleteAction(certificatesigningrequestsResource, name), &certificates.CertificateSigningRequest{})
	return err
}

func (c *FakeCertificateSigningRequests) DeleteCollection(options *api.DeleteOptions, listOptions api.ListOptions) error {
	action := core.NewRootDeleteCollectionAction(certificatesigningrequestsResource, listOptions)

	_, err := c.Fake.Invokes(action, &certificates.CertificateSigningRequestList{})
	return err
}

func (c *FakeCertificateSigningRequests) Get(name string, options v1.GetOptions) (result *certificates.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootGetAction(certificatesigningrequestsResource, name), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) List(opts api.ListOptions) (result *certificates.CertificateSigningRequestList, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootListAction(certificatesigningrequestsResource, opts), &certificates.CertificateSigningRequestList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := core.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &certificates.CertificateSigningRequestList{}
	for _, item := range obj.(*certificates.CertificateSigningRequestList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested certificateSigningRequests.
func (c *FakeCertificateSigningRequests) Watch(opts api.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(core.NewRootWatchAction(certificatesigningrequestsResource, opts))
}

// Patch applies the patch and returns the patched certificateSigningRequest.
func (c *FakeCertificateSigningRequests) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *certificates.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(core.NewRootPatchSubresourceAction(certificatesigningrequestsResource, name, data, subresources...), &certificates.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), err
}
