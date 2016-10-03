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
	v1alpha1 "k8s.io/client-go/1.4/kubernetes/typed/certificates/v1alpha1"
	rest "k8s.io/client-go/1.4/rest"
	testing "k8s.io/client-go/1.4/testing"
)

type FakeCertificates struct {
	*testing.Fake
}

func (c *FakeCertificates) CertificateSigningRequests() v1alpha1.CertificateSigningRequestInterface {
	return &FakeCertificateSigningRequests{c}
}

// GetRESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeCertificates) GetRESTClient() *rest.RESTClient {
	return nil
}
