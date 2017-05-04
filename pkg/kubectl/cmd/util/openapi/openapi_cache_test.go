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
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

var _ = Describe("When reading openAPIData", func() {
	var tmpDir string
	var err error
	var client *fakeOpenAPIClient
	var instance *openapi.CachingOpenAPIClient
	var expectedData *openapi.Resources

	BeforeEach(func() {
		tmpDir, err = ioutil.TempDir("", "openapi_cache_test")
		Expect(err).To(BeNil())
		client = &fakeOpenAPIClient{}
		instance = openapi.NewCachingOpenAPIClient(client, "v1.6", tmpDir)

		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())

		expectedData, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	AfterEach(func() {
		os.RemoveAll(tmpDir)
	})

	It("should write to the cache", func() {
		By("getting the live openapi spec from the server")
		result, err := instance.OpenAPIData()
		Expect(err).To(BeNil())
		expectEqual(result, expectedData)
		Expect(client.calls).To(Equal(1))

		By("writing the live openapi spec to a local cache file")
		names, err := getFilenames(tmpDir)
		Expect(err).To(BeNil())
		Expect(names).To(ConsistOf("v1.6"))

		names, err = getFilenames(filepath.Join(tmpDir, "v1.6"))
		Expect(err).To(BeNil())
		Expect(names).To(HaveLen(1))
		clientVersion := names[0]

		names, err = getFilenames(filepath.Join(tmpDir, "v1.6", clientVersion))
		Expect(err).To(BeNil())
		Expect(names).To(ContainElement("openapi_cache"))
	})

	It("should read from the cache", func() {
		// First call should use the client
		result, err := instance.OpenAPIData()
		Expect(err).To(BeNil())
		expectEqual(result, expectedData)
		Expect(client.calls).To(Equal(1))

		// Second call shouldn't use the client
		result, err = instance.OpenAPIData()
		Expect(err).To(BeNil())
		expectEqual(result, expectedData)
		Expect(client.calls).To(Equal(1))

		names, err := getFilenames(tmpDir)
		Expect(err).To(BeNil())
		Expect(names).To(ConsistOf("v1.6"))
	})

	It("propagate errors that are encountered", func() {
		// Expect an error
		client.err = fmt.Errorf("expected error")
		result, err := instance.OpenAPIData()
		Expect(err.Error()).To(Equal(client.err.Error()))
		Expect(result).To(BeNil())
		Expect(client.calls).To(Equal(1))

		// No cache file is written
		files, err := ioutil.ReadDir(tmpDir)
		Expect(err).To(BeNil())
		Expect(files).To(HaveLen(0))

		// Client error is not cached
		result, err = instance.OpenAPIData()
		Expect(err.Error()).To(Equal(client.err.Error()))
		Expect(result).To(BeNil())
		Expect(client.calls).To(Equal(2))
	})
})

var _ = Describe("Reading openAPIData", func() {
	var tmpDir string
	var serverVersion string
	var cacheDir string

	BeforeEach(func() {
		var err error
		tmpDir, err = ioutil.TempDir("", "openapi_cache_test")
		Expect(err).To(BeNil())
	})

	AfterEach(func() {
		os.RemoveAll(tmpDir)
	})

	// Set the serverVersion to empty
	Context("when the server version is empty", func() {
		BeforeEach(func() {
			serverVersion = ""
			cacheDir = tmpDir
		})
		It("should not cache the result", func() {
			client := &fakeOpenAPIClient{}

			instance := openapi.NewCachingOpenAPIClient(client, serverVersion, cacheDir)

			d, err := data.OpenAPISchema()
			Expect(err).To(BeNil())

			expectedData, err := openapi.NewOpenAPIData(d)
			Expect(err).To(BeNil())

			By("getting the live openapi schema")
			result, err := instance.OpenAPIData()
			Expect(err).To(BeNil())
			expectEqual(result, expectedData)
			Expect(client.calls).To(Equal(1))

			files, err := ioutil.ReadDir(tmpDir)
			Expect(err).To(BeNil())
			Expect(files).To(HaveLen(0))
		})
	})

	Context("when the cache directory is empty", func() {
		BeforeEach(func() {
			serverVersion = "v1.6"
			cacheDir = ""
		})
		It("should not cache the result", func() {
			client := &fakeOpenAPIClient{}

			instance := openapi.NewCachingOpenAPIClient(client, serverVersion, cacheDir)

			d, err := data.OpenAPISchema()
			Expect(err).To(BeNil())

			expectedData, err := openapi.NewOpenAPIData(d)
			Expect(err).To(BeNil())

			By("getting the live openapi schema")
			result, err := instance.OpenAPIData()
			Expect(err).To(BeNil())
			expectEqual(result, expectedData)
			Expect(client.calls).To(Equal(1))

			files, err := ioutil.ReadDir(tmpDir)
			Expect(err).To(BeNil())
			Expect(files).To(HaveLen(0))
		})
	})
})

// Test Utils
func getFilenames(path string) ([]string, error) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, n := range files {
		result = append(result, n.Name())
	}
	return result, nil
}

func expectEqual(a *openapi.Resources, b *openapi.Resources) {
	Expect(a.NameToDefinition).To(HaveLen(len(b.NameToDefinition)))
	for k, v := range a.NameToDefinition {
		Expect(v).To(Equal(b.NameToDefinition[k]),
			fmt.Sprintf("Names for GVK do not match %v", k))
	}
	Expect(a.GroupVersionKindToName).To(HaveLen(len(b.GroupVersionKindToName)))
	for k, v := range a.GroupVersionKindToName {
		Expect(v).To(Equal(b.GroupVersionKindToName[k]),
			fmt.Sprintf("Values for name do not match %v", k))
	}
}

type fakeOpenAPIClient struct {
	calls int
	err   error
}

func (f *fakeOpenAPIClient) OpenAPISchema() (*spec.Swagger, error) {
	f.calls = f.calls + 1

	if f.err != nil {
		return nil, f.err
	}

	return data.OpenAPISchema()
}

// Test utils
var data apiData

type apiData struct {
	sync.Once
	data *spec.Swagger
	err  error
}

func (d *apiData) OpenAPISchema() (*spec.Swagger, error) {
	d.Do(func() {
		// Get the path to the swagger.json file
		wd, err := os.Getwd()
		if err != nil {
			d.err = err
			return
		}

		abs, err := filepath.Abs(wd)
		if err != nil {
			d.err = err
			return
		}

		root := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(abs)))))
		specpath := filepath.Join(root, "api", "openapi-spec", "swagger.json")
		_, err = os.Stat(specpath)
		if err != nil {
			d.err = err
			return
		}
		// Load the openapi document
		doc, err := loads.Spec(specpath)
		if err != nil {
			d.err = err
			return
		}

		d.data = doc.Spec()
	})
	return d.data, d.err
}
