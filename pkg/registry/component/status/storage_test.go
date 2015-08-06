/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package status

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	mock_probe_http "k8s.io/kubernetes/pkg/probe/http/mocks"
	mock_component "k8s.io/kubernetes/pkg/registry/component/mocks"
	mock_db_storage "k8s.io/kubernetes/pkg/storage/mocks"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func newResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		Body:       ioutil.NopCloser(bytes.NewBufferString(body)),
		StatusCode: statusCode,
	}
}

func newComponentStatus(name string, status api.ConditionStatus, msg string, err string) *api.ComponentStatus {
	return &api.ComponentStatus{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Conditions: []api.ComponentCondition{
			{
				Type:    api.ComponentHealthy,
				Status:  status,
				Message: msg,
				Error:   err,
			},
		},
	}
}

var _ = Describe("REST", func() {
	var (
		mockRegistry   *mock_component.MockRegistry
		mockDatabase   *mock_db_storage.MockInterface
		mockHTTPGetter *mock_probe_http.MockHTTPGetter

		storage rest.ReadableStorage
	)

	BeforeEach(func() {
		mockRegistry = &mock_component.MockRegistry{}
		mockDatabase = &mock_db_storage.MockInterface{}
		mockHTTPGetter = &mock_probe_http.MockHTTPGetter{}
		storage = NewStorage(mockRegistry, mockDatabase, mockHTTPGetter)
	})

	Describe("List", func() {

		Context("When one component is registered", func() {
			var componentName = "scheduler-xxxxx"

			BeforeEach(func() {
				// one component
				componentList := &api.ComponentList{
					Items: []api.Component{
						{
							ObjectMeta: api.ObjectMeta{
								Name: componentName,
							},
							Type: api.ComponentScheduler,
							URL:  "http://nowhere.tld",
						},
					},
				}
				mockRegistry.ListComponentsReturns(componentList, nil)

				// zero db nodes
				mockDatabase.BackendsReturns([]string{})
			})

			Context("When component is healthy", func() {
				var body = "ok"

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(newResponse(http.StatusOK, body), nil)
				})

				It("Returns success", func() {
					result, err := storage.List(api.NewContext(), labels.Everything(), fields.Everything())
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(&api.ComponentStatusList{
						Items: []api.ComponentStatus{
							*newComponentStatus(componentName, api.ConditionTrue, body, "nil"),
						},
					}))
				})
			})

			Context("When component is unhealthy", func() {
				var body = "unhealthy http status code: 500 ()"

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(newResponse(http.StatusInternalServerError, body), nil)
				})

				It("Returns failure", func() {
					result, err := storage.List(api.NewContext(), labels.Everything(), fields.Everything())
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(&api.ComponentStatusList{
						Items: []api.ComponentStatus{
							*newComponentStatus(componentName, api.ConditionFalse, body, "nil"),
						},
					}))
				})
			})

			//TODO: api.ConditionUnknown used to be returned for probe errors. Do we need to be reverse compatible or was that a bug?
			Context("When probe errors", func() {
				var probeError = errors.New("probe-error")

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(nil, probeError)
				})

				It("Returns failure", func() {
					result, err := storage.List(api.NewContext(), labels.Everything(), fields.Everything())
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(&api.ComponentStatusList{
						Items: []api.ComponentStatus{
							*newComponentStatus(componentName, api.ConditionFalse, probeError.Error(), "nil"),
						},
					}))
				})
			})
		})

	})

	Describe("Get", func() {

		Context("When one component is registered", func() {
			var componentName = "scheduler-xxxxx"

			BeforeEach(func() {
				// one component
				component := &api.Component{
					ObjectMeta: api.ObjectMeta{
						Name: componentName,
					},
					Type: api.ComponentScheduler,
					URL:  "http://nowhere.tld",
				}
				mockRegistry.GetComponentReturns(component, nil)

				// zero db nodes
				mockDatabase.BackendsReturns([]string{})
			})

			Context("When component is healthy", func() {
				var body = "ok"

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(newResponse(http.StatusOK, body), nil)
				})

				It("Returns success", func() {
					result, err := storage.Get(api.NewContext(), componentName)
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(newComponentStatus(componentName, api.ConditionTrue, body, "nil")))
				})
			})

			Context("When component is unhealthy", func() {
				var body = "unhealthy http status code: 500 ()"

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(newResponse(http.StatusInternalServerError, body), nil)
				})

				It("Returns failure", func() {
					result, err := storage.Get(api.NewContext(), componentName)
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(newComponentStatus(componentName, api.ConditionFalse, body, "nil")))
				})
			})

			//TODO: api.ConditionUnknown used to be returned for probe errors. Do we need to be reverse compatible or was that a bug?
			Context("When probe errors", func() {
				var probeError = errors.New("probe-error")

				BeforeEach(func() {
					mockHTTPGetter.GetReturns(nil, probeError)
				})

				It("Returns failure", func() {
					result, err := storage.Get(api.NewContext(), componentName)
					Expect(err).ToNot(HaveOccurred())

					Expect(result).To(Equal(newComponentStatus(componentName, api.ConditionFalse, probeError.Error(), "nil")))
				})
			})

			Context("When provided name does not match a registered component", func() {
				var getComponentError = errors.New("get-component-error")

				BeforeEach(func() {
					mockRegistry.GetComponentReturns(nil, getComponentError)
				})

				It("Returns an error", func() {
					_, err := storage.Get(api.NewContext(), componentName)
					Expect(err).To(HaveOccurred())
					Expect(err.Error()).To(ContainSubstring("failed to retrieve component status: %s", getComponentError.Error()))
				})
			})
		})

	})
})
