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

package openapi_test

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	oapi "k8s.io/apimachinery/pkg/util/openapi"
	tst "k8s.io/apimachinery/pkg/util/openapi/testing"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"

	"github.com/googleapis/gnostic/OpenAPIv2"
)

// FakeClient implements a dummy OpenAPISchemaInterface that uses the
// fake OpenAPI schema given as a parameter, and count the number of
// call to the function.
type FakeClient struct {
	Calls int
	Err   error

	fake tst.Getter
}

// NewFakeClient creates a new FakeClient from the given Fake.
func NewFakeClient(f tst.Getter) *FakeClient {
	return &FakeClient{fake: f}
}

// OpenAPISchema returns a OpenAPI Document as returned by the fake, but
// it also counts the number of calls.
func (f *FakeClient) OpenAPISchema() (*openapi_v2.Document, error) {
	f.Calls = f.Calls + 1

	if f.Err != nil {
		return nil, f.Err
	}

	return f.fake.OpenAPISchema()
}

var _ = Describe("Getting the Resources", func() {
	var client *FakeClient
	var expectedData oapi.Resources
	var instance openapi.Getter

	BeforeEach(func() {
		client = NewFakeClient(tst.Empty{})

		var err error
		expectedData, err = oapi.NewOpenAPIData(&openapi_v2.Document{})
		Expect(err).To(BeNil())

		instance = openapi.NewOpenAPIGetter(client)
	})

	Context("when the server returns a successful result", func() {
		It("should return the same data for multiple calls", func() {
			Expect(client.Calls).To(Equal(0))

			result, err := instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			Expect(client.Calls).To(Equal(1))

			result, err = instance.Get()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			// No additional client calls expected
			Expect(client.Calls).To(Equal(1))
		})
	})

	Context("when the server returns an unsuccessful result", func() {
		It("should return the same instance for multiple calls.", func() {
			Expect(client.Calls).To(Equal(0))

			client.Err = fmt.Errorf("expected error")
			_, err := instance.Get()
			Expect(err).To(Equal(client.Err))
			Expect(client.Calls).To(Equal(1))

			_, err = instance.Get()
			Expect(err).To(Equal(client.Err))
			// No additional client calls expected
			Expect(client.Calls).To(Equal(1))
		})
	})
})
