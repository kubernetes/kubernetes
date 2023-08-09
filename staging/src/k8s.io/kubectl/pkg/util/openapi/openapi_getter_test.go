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

	openapi_v2 "github.com/google/gnostic/openapiv2"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/kubectl/pkg/util/openapi"
)

// FakeCounter returns a "null" document and the specified error. It
// also counts how many times the OpenAPISchema method has been called.
type FakeCounter struct {
	Calls int
	Err   error
}

func (f *FakeCounter) OpenAPISchema() (*openapi_v2.Document, error) {
	f.Calls = f.Calls + 1
	return nil, f.Err
}

var _ = Describe("Getting the Resources", func() {
	var client FakeCounter
	var instance *openapi.CachedOpenAPIParser
	var expectedData openapi.Resources

	BeforeEach(func() {
		client = FakeCounter{}
		instance = openapi.NewOpenAPIParser(openapi.NewOpenAPIGetter(&client))
		var err error
		expectedData, err = openapi.NewOpenAPIData(nil)
		Expect(err).To(BeNil())
	})

	Context("when the server returns a successful result", func() {
		It("should return the same data for multiple calls", func() {
			Expect(client.Calls).To(Equal(0))

			result, err := instance.Parse()
			Expect(err).To(BeNil())
			Expect(result).To(Equal(expectedData))
			Expect(client.Calls).To(Equal(1))

			result, err = instance.Parse()
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
			_, err := instance.Parse()
			Expect(err).To(Equal(client.Err))
			Expect(client.Calls).To(Equal(1))

			_, err = instance.Parse()
			Expect(err).To(Equal(client.Err))
			// No additional client calls expected
			Expect(client.Calls).To(Equal(1))
		})
	})
})
