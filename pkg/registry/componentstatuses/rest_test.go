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

package componentstatuses

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	mock_storage "k8s.io/kubernetes/pkg/storage/mocks"

	. "github.com/onsi/gomega"
)

type fakeRoundTripper struct {
	err  error
	resp func() *http.Response
	url  string
}

func (f *fakeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	f.url = req.URL.String()
	return f.resp(), f.err
}

type testResponse struct {
	code int
	data string
	err  error
}

func NewTestREST(resp testResponse) rest.ReadableStorage {
	databaseStorage := &mock_storage.MockInterface{}
	rt := &fakeRoundTripper{
		err: resp.err,
		resp: func() *http.Response {
			return &http.Response{
				Body:       ioutil.NopCloser(bytes.NewBufferString(resp.data)),
				StatusCode: resp.code,
			}
		},
	}
	return NewStorage(databaseStorage, rt)
}

func createTestStatus(name string, status api.ConditionStatus, msg string, err string) *api.ComponentStatuses {
	retVal := &api.ComponentStatuses{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: status, Message: msg, Error: err},
		},
	}
	retVal.Name = name
	return retVal
}

func TestList_NoError(t *testing.T) {
	RegisterTestingT(t)
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	Expect(err).ToNot(HaveOccurred())
	Expect(got).To(Equal(&api.ComponentStatusesList{
		Items: []api.ComponentStatuses{
			*(createTestStatus("controller-manager", api.ConditionTrue, "ok", "nil")),
			*(createTestStatus("scheduler", api.ConditionTrue, "ok", "nil")),
		},
	}))
}

func TestList_FailedCheck(t *testing.T) {
	RegisterTestingT(t)
	r := NewTestREST(testResponse{code: 500, data: ""})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	Expect(err).ToNot(HaveOccurred())
	Expect(got).To(Equal(&api.ComponentStatusesList{
		Items: []api.ComponentStatuses{
			*(createTestStatus("controller-manager", api.ConditionFalse, "", "unhealthy http status code: 500 ()")),
			*(createTestStatus("scheduler", api.ConditionFalse, "", "unhealthy http status code: 500 ()")),
		},
	}))
}

func TestList_UnknownError(t *testing.T) {
	RegisterTestingT(t)
	r := NewTestREST(testResponse{code: 500, data: "", err: fmt.Errorf("fizzbuzz error")})
	got, err := r.List(api.NewContext(), labels.Everything(), fields.Everything())
	Expect(err).ToNot(HaveOccurred())
	Expect(got).To(Equal(&api.ComponentStatusesList{
		Items: []api.ComponentStatuses{
			*(createTestStatus("controller-manager", api.ConditionUnknown, "", "Get http://127.0.0.1:10252/healthz: fizzbuzz error")),
			*(createTestStatus("scheduler", api.ConditionUnknown, "", "Get http://127.0.0.1:10251/healthz: fizzbuzz error")),
		},
	}))
}

func TestGet_NoError(t *testing.T) {
	RegisterTestingT(t)
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	got, err := r.Get(api.NewContext(), "controller-manager")
	Expect(err).ToNot(HaveOccurred())
	Expect(got).To(Equal(createTestStatus("controller-manager", api.ConditionTrue, "ok", "nil")))
}

func TestGet_BadName(t *testing.T) {
	RegisterTestingT(t)
	r := NewTestREST(testResponse{code: 200, data: "ok"})
	_, err := r.Get(api.NewContext(), "invalidname")
	Expect(err).To(HaveOccurred())
	Expect(err.Error()).To(ContainSubstring("Component not found: invalidname"))
}
