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

	"gopkg.in/yaml.v2"

	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

// Test utils
var data apiData

type apiData struct {
	sync.Once
	data *openapi_v2.Document
	err  error
}

func (d *apiData) OpenAPISchema() (*openapi_v2.Document, error) {
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
		spec, err := ioutil.ReadFile(specpath)
		if err != nil {
			d.err = err
			return
		}
		var info yaml.MapSlice
		err = yaml.Unmarshal(spec, &info)
		if err != nil {
			d.err = err
			return
		}
		d.data, d.err = openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	})
	return d.data, d.err
}

type fakeOpenAPIClient struct {
	calls int
	err   error
}

func (f *fakeOpenAPIClient) OpenAPISchema() (*openapi_v2.Document, error) {
	f.calls = f.calls + 1

	if f.err != nil {
		return nil, f.err
	}

	return data.OpenAPISchema()
}

var _ = Describe("Getting the Resources", func() {
	var client *tst.FakeClient
	var expectedData openapi.Resources
	var instance openapi.Getter

	BeforeEach(func() {
		client = tst.NewFakeClient(&fakeSchema)
		d, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())

		expectedData, err = openapi.NewOpenAPIData(d)
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
