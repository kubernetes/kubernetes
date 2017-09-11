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

package endpoints

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	genericapitesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/registry/rest"
)

type fakeAuditSink struct {
	lock   sync.Mutex
	events []*auditinternal.Event
}

func (s *fakeAuditSink) ProcessEvents(evs ...*auditinternal.Event) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.events = append(s.events, evs...)
}

func (s *fakeAuditSink) Events() []*auditinternal.Event {
	s.lock.Lock()
	defer s.lock.Unlock()
	return append([]*auditinternal.Event{}, s.events...)
}

func TestAudit(t *testing.T) {
	type eventCheck func(events []*auditinternal.Event) error

	// fixtures
	simpleFoo := &genericapitesting.Simple{Other: "foo"}
	simpleFooJSON, _ := runtime.Encode(testCodec, simpleFoo)

	simpleCPrime := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "other"},
		Other:      "bla",
	}
	simpleCPrimeJSON, _ := runtime.Encode(testCodec, simpleCPrime)

	// event checks
	noRequestBody := func(i int) eventCheck {
		return func(events []*auditinternal.Event) error {
			if events[i].RequestObject == nil {
				return nil
			}
			return fmt.Errorf("expected RequestBody to be nil, got non-nill '%s'", events[i].RequestObject.Raw)
		}
	}
	requestBodyIs := func(i int, text string) eventCheck {
		return func(events []*auditinternal.Event) error {
			if events[i].RequestObject == nil {
				if text != "" {
					return fmt.Errorf("expected RequestBody %q, got <nil>", text)
				}
				return nil
			}
			if string(events[i].RequestObject.Raw) != text {
				return fmt.Errorf("expected RequestBody %q, got %q", text, string(events[i].RequestObject.Raw))
			}
			return nil
		}
	}
	requestBodyMatches := func(i int, pattern string) eventCheck {
		return func(events []*auditinternal.Event) error {
			if matched, _ := regexp.Match(pattern, events[i].RequestObject.Raw); !matched {
				return fmt.Errorf("expected RequestBody to match %q, but didn't: %q", pattern, string(events[i].RequestObject.Raw))
			}
			return nil
		}
	}
	noResponseBody := func(i int) eventCheck {
		return func(events []*auditinternal.Event) error {
			if events[i].ResponseObject == nil {
				return nil
			}
			return fmt.Errorf("expected ResponseBody to be nil, got non-nill '%s'", events[i].ResponseObject.Raw)
		}
	}
	responseBodyMatches := func(i int, pattern string) eventCheck {
		return func(events []*auditinternal.Event) error {
			if matched, _ := regexp.Match(pattern, events[i].ResponseObject.Raw); !matched {
				return fmt.Errorf("expected ResponseBody to match %q, but didn't: %q", pattern, string(events[i].ResponseObject.Raw))
			}
			return nil
		}
	}

	for _, test := range []struct {
		desc   string
		req    func(server string) (*http.Request, error)
		linker runtime.SelfLinker
		code   int
		events int
		checks []eventCheck
	}{
		{
			"get",
			func(server string) (*http.Request, error) {
				return http.NewRequest("GET", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/other/simple/c", bytes.NewBuffer(simpleFooJSON))
			},
			selfLinker,
			200,
			2,
			[]eventCheck{
				noRequestBody(0),
				responseBodyMatches(0, `{.*"name":"c".*}`),
			},
		},
		{
			"list",
			func(server string) (*http.Request, error) {
				return http.NewRequest("GET", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/other/simple?labelSelector=a%3Dfoobar", nil)
			},
			&setTestSelfLinker{
				t:           t,
				expectedSet: "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/other/simple",
				namespace:   "other",
			},
			200,
			2,
			[]eventCheck{
				noRequestBody(0),
				responseBodyMatches(0, `{.*"name":"a".*"name":"b".*}`),
			},
		},
		{
			"create",
			func(server string) (*http.Request, error) {
				return http.NewRequest("POST", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple", bytes.NewBuffer(simpleFooJSON))
			},
			selfLinker,
			201,
			2,
			[]eventCheck{
				requestBodyIs(0, string(simpleFooJSON)),
				responseBodyMatches(0, `{.*"foo".*}`),
			},
		},
		{
			"not-allowed-named-create",
			func(server string) (*http.Request, error) {
				return http.NewRequest("POST", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/named", bytes.NewBuffer(simpleFooJSON))
			},
			selfLinker,
			405,
			2,
			[]eventCheck{
				noRequestBody(0),  // the 405 is thrown long before the create handler would be executed
				noResponseBody(0), // the 405 is thrown long before the create handler would be executed
			},
		},
		{
			"delete",
			func(server string) (*http.Request, error) {
				return http.NewRequest("DELETE", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/a", nil)
			},
			selfLinker,
			200,
			2,
			[]eventCheck{
				noRequestBody(0),
				responseBodyMatches(0, `{.*"kind":"Status".*"status":"Success".*}`),
			},
		},
		{
			"delete-with-options-in-body",
			func(server string) (*http.Request, error) {
				return http.NewRequest("DELETE", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/a", bytes.NewBuffer([]byte(`{"kind":"DeleteOptions"}`)))
			},
			selfLinker,
			200,
			2,
			[]eventCheck{
				requestBodyMatches(0, "DeleteOptions"),
				responseBodyMatches(0, `{.*"kind":"Status".*"status":"Success".*}`),
			},
		},
		{
			"update",
			func(server string) (*http.Request, error) {
				return http.NewRequest("PUT", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/other/simple/c", bytes.NewBuffer(simpleCPrimeJSON))
			},
			selfLinker,
			200,
			2,
			[]eventCheck{
				requestBodyIs(0, string(simpleCPrimeJSON)),
				responseBodyMatches(0, `{.*"bla".*}`),
			},
		},
		{
			"update-wrong-namespace",
			func(server string) (*http.Request, error) {
				return http.NewRequest("PUT", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/c", bytes.NewBuffer(simpleCPrimeJSON))
			},
			selfLinker,
			400,
			2,
			[]eventCheck{
				requestBodyIs(0, string(simpleCPrimeJSON)),
				responseBodyMatches(0, `"Status".*"status":"Failure".*"code":400}`),
			},
		},
		{
			"patch",
			func(server string) (*http.Request, error) {
				req, _ := http.NewRequest("PATCH", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/other/simple/c", bytes.NewReader([]byte(`{"labels":{"foo":"bar"}}`)))
				req.Header.Set("Content-Type", "application/merge-patch+json; charset=UTF-8")
				return req, nil
			},
			&setTestSelfLinker{
				t:           t,
				expectedSet: "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/other/simple/c",
				name:        "c",
				namespace:   "other",
			},
			200,
			2,
			[]eventCheck{
				requestBodyIs(0, `{"labels":{"foo":"bar"}}`),
				responseBodyMatches(0, `"name":"c".*"labels":{"foo":"bar"}`),
			},
		},
		{
			"watch",
			func(server string) (*http.Request, error) {
				return http.NewRequest("GET", server+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/other/simple?watch=true", nil)
			},
			&setTestSelfLinker{
				t:           t,
				expectedSet: "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/other/simple",
				namespace:   "other",
			},
			200,
			3,
			[]eventCheck{
				noRequestBody(0),
				noResponseBody(0),
			},
		},
	} {
		sink := &fakeAuditSink{}
		handler := handleInternal(map[string]rest.Storage{
			"simple": &SimpleRESTStorage{
				list: []genericapitesting.Simple{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "other"},
						Other:      "foo",
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "other"},
						Other:      "foo",
					},
				},
				item: genericapitesting.Simple{
					ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "other", UID: "uid"},
					Other:      "foo",
				},
			},
		}, admissionControl, selfLinker, sink)

		server := httptest.NewServer(handler)
		defer server.Close()
		client := http.Client{Timeout: 2 * time.Second}

		req, err := test.req(server.URL)
		if err != nil {
			t.Errorf("[%s] error creating the request: %v", test.desc, err)
		}

		response, err := client.Do(req)
		if err != nil {
			t.Errorf("[%s] error: %v", test.desc, err)
		}

		if response.StatusCode != test.code {
			t.Errorf("[%s] expected http code %d, got %#v", test.desc, test.code, response)
		}

		// close body because the handler might block in Flush, unable to send the remaining event.
		response.Body.Close()

		// wait for events to arrive, at least the given number in the test
		events := []*auditinternal.Event{}
		err = wait.Poll(50*time.Millisecond, wait.ForeverTestTimeout, wait.ConditionFunc(func() (done bool, err error) {
			events = sink.Events()
			return len(events) >= test.events, nil
		}))
		if err != nil {
			t.Errorf("[%s] timeout waiting for events", test.desc)
		}

		if got := len(events); got != test.events {
			t.Errorf("[%s] expected %d audit events, got %d", test.desc, test.events, got)
		} else {
			for i, check := range test.checks {
				err := check(events)
				if err != nil {
					t.Errorf("[%s,%d] %v", test.desc, i, err)
				}
			}
		}

		if len(events) > 0 {
			status := events[len(events)-1].ResponseStatus
			if status == nil {
				t.Errorf("[%s] expected non-nil ResponseStatus in last event", test.desc)
			} else if int(status.Code) != test.code {
				t.Errorf("[%s] expected ResponseStatus.Code=%d, got %d", test.desc, test.code, status.Code)
			}
		}
	}
}
