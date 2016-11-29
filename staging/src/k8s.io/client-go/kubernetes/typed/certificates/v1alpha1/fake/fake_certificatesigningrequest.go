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

package fake

import (
	api "k8s.io/client-go/pkg/api"
	v1 "k8s.io/client-go/pkg/api/v1"
	v1alpha1 "k8s.io/client-go/pkg/apis/certificates/v1alpha1"
	labels "k8s.io/client-go/pkg/labels"
	schema "k8s.io/client-go/pkg/runtime/schema"
	watch "k8s.io/client-go/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeCertificateSigningRequests implements CertificateSigningRequestInterface
type FakeCertificateSigningRequests struct {
	Fake *FakeCertificatesV1alpha1
}

var certificatesigningrequestsResource = schema.GroupVersionResource{Group: "certificates.k8s.io", Version: "v1alpha1", Resource: "certificatesigningrequests"}

func (c *FakeCertificateSigningRequests) Create(certificateSigningRequest *v1alpha1.CertificateSigningRequest) (result *v1alpha1.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootCreateAction(certificatesigningrequestsResource, certificateSigningRequest), &v1alpha1.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) Update(certificateSigningRequest *v1alpha1.CertificateSigningRequest) (result *v1alpha1.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateAction(certificatesigningrequestsResource, certificateSigningRequest), &v1alpha1.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) UpdateStatus(certificateSigningRequest *v1alpha1.CertificateSigningRequest) (*v1alpha1.CertificateSigningRequest, error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootUpdateSubresourceAction(certificatesigningrequestsResource, "status", certificateSigningRequest), &v1alpha1.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) Delete(name string, options *v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewRootDeleteAction(certificatesigningrequestsResource, name), &v1alpha1.CertificateSigningRequest{})
	return err
}

func (c *FakeCertificateSigningRequests) DeleteCollection(options *v1.DeleteOptions, listOptions v1.ListOptions) error {
	action := testing.NewRootDeleteCollectionAction(certificatesigningrequestsResource, listOptions)

	_, err := c.Fake.Invokes(action, &v1alpha1.CertificateSigningRequestList{})
	return err
}

func (c *FakeCertificateSigningRequests) Get(name string) (result *v1alpha1.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootGetAction(certificatesigningrequestsResource, name), &v1alpha1.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CertificateSigningRequest), err
}

func (c *FakeCertificateSigningRequests) List(opts v1.ListOptions) (result *v1alpha1.CertificateSigningRequestList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootListAction(certificatesigningrequestsResource, opts), &v1alpha1.CertificateSigningRequestList{})
	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.CertificateSigningRequestList{}
	for _, item := range obj.(*v1alpha1.CertificateSigningRequestList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested certificateSigningRequests.
func (c *FakeCertificateSigningRequests) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewRootWatchAction(certificatesigningrequestsResource, opts))
}

// Patch applies the patch and returns the patched certificateSigningRequest.
func (c *FakeCertificateSigningRequests) Patch(name string, pt api.PatchType, data []byte, subresources ...string) (result *v1alpha1.CertificateSigningRequest, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewRootPatchSubresourceAction(certificatesigningrequestsResource, name, data, subresources...), &v1alpha1.CertificateSigningRequest{})
	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.CertificateSigningRequest), err
}
