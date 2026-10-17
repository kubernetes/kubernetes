/*
Copyright The Kubernetes Authors.

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
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	restful "github.com/emicklei/go-restful/v3"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/admission"
	genericapitesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/registry/rest"
)

// equivalentsStorage declares admission equivalents for the endpoint it is installed at.
type equivalentsStorage struct {
	*SimpleRESTStorageWithDeleteCollection
	eqs []admission.Equivalent
}

func (s *equivalentsStorage) AdmissionEquivalents() []admission.Equivalent {
	return s.eqs
}

// DeleteCollection calls deleteValidation, which is where the handler runs admission, unlike the
// embedded fixture.
func (s *equivalentsStorage) DeleteCollection(ctx context.Context, deleteValidation rest.ValidateObjectFunc, _ *metav1.DeleteOptions, _ *metainternalversion.ListOptions) (runtime.Object, error) {
	if err := deleteValidation(ctx, &s.item); err != nil {
		return nil, err
	}
	return &genericapitesting.SimpleList{}, nil
}

type equivalentsConnecterStorage struct {
	*ConnecterRESTStorage
	eqs []admission.Equivalent
}

func (s *equivalentsConnecterStorage) AdmissionEquivalents() []admission.Equivalent {
	return s.eqs
}

var (
	_ rest.AdmissionEquivalentsProvider = &equivalentsStorage{}
	_ rest.AdmissionEquivalentsProvider = &equivalentsConnecterStorage{}
)

type admissionCall struct {
	phase       string // "mutate" or "validate"
	operation   admission.Operation
	subresource string
	eqs         []admission.Equivalent
}

// equivalentsRecorder records the admission equivalents visible to admission for every call.
type equivalentsRecorder struct {
	lock  sync.Mutex
	calls []admissionCall
}

var (
	_ admission.MutationInterface   = &equivalentsRecorder{}
	_ admission.ValidationInterface = &equivalentsRecorder{}
)

func (r *equivalentsRecorder) Handles(admission.Operation) bool { return true }

func (r *equivalentsRecorder) record(phase string, a admission.Attributes, o admission.ObjectInterfaces) {
	r.lock.Lock()
	defer r.lock.Unlock()
	var eqs []admission.Equivalent
	if g, ok := o.(admission.EquivalentsGetter); ok {
		eqs = g.GetAdmissionEquivalents()
	}
	r.calls = append(r.calls, admissionCall{phase: phase, operation: a.GetOperation(), subresource: a.GetSubresource(), eqs: eqs})
}

func (r *equivalentsRecorder) Admit(_ context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	r.record("mutate", a, o)
	return nil
}

func (r *equivalentsRecorder) Validate(_ context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	r.record("validate", a, o)
	return nil
}

func (r *equivalentsRecorder) reset() []admissionCall {
	r.lock.Lock()
	defer r.lock.Unlock()
	calls := r.calls
	r.calls = nil
	return calls
}

// TestAdmissionEquivalentsReachAdmission verifies that every handler that calls admission passes
// the endpoint's declared admission equivalents, so that admission plugins can enforce coverage.
func TestAdmissionEquivalentsReachAdmission(t *testing.T) {
	baseEqs := []admission.Equivalent{{Subresource: "turbo", Operations: []admission.Operation{admission.Update}}}
	turboEqs := []admission.Equivalent{{Subresource: ""}}
	connectEqs := []admission.Equivalent{{Subresource: "", Operations: []admission.Operation{admission.Update}}}

	newSimpleStorage := func(uid string) *SimpleRESTStorageWithDeleteCollection {
		return &SimpleRESTStorageWithDeleteCollection{SimpleRESTStorage{
			item: genericapitesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "id", Namespace: "default", UID: types.UID(uid)}, Other: "foo"},
		}}
	}

	recorder := &equivalentsRecorder{}
	storage := map[string]rest.Storage{
		"simples":       &equivalentsStorage{SimpleRESTStorageWithDeleteCollection: newSimpleStorage("uid"), eqs: baseEqs},
		"simples/turbo": &equivalentsStorage{SimpleRESTStorageWithDeleteCollection: newSimpleStorage("uid"), eqs: turboEqs},
		// The fixture treats an update of an object without a UID as a create.
		"simples/fresh":   &equivalentsStorage{SimpleRESTStorageWithDeleteCollection: newSimpleStorage(""), eqs: turboEqs},
		"simples/connect": &equivalentsConnecterStorage{ConnecterRESTStorage: &ConnecterRESTStorage{connectHandler: &OutputConnect{response: "ok"}}, eqs: connectEqs},
		"others":          &SimpleRESTStorage{item: genericapitesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "id", Namespace: "default", UID: "uid"}}},
	}
	server := httptest.NewServer(handleInternal(storage, recorder, nil))
	defer server.Close()

	simpleBody := func(name string) []byte {
		return fmt.Appendf(nil, `{"kind":"Simple","apiVersion":"test.group/version","metadata":{"name":%q,"namespace":"default"},"other":"bar"}`, name)
	}
	base := server.URL + "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/default"

	testcases := []struct {
		name        string
		method      string
		path        string
		body        []byte
		contentType string

		expectOp   admission.Operation
		expectSub  string
		expectEqs  []admission.Equivalent
		mutateOnly bool
	}{
		{name: "create", method: http.MethodPost, path: "/simples", body: simpleBody(""), expectOp: admission.Create, expectEqs: baseEqs},
		{name: "update", method: http.MethodPut, path: "/simples/id/turbo", body: simpleBody("id"), expectOp: admission.Update, expectSub: "turbo", expectEqs: turboEqs},
		{
			name:      "update as create",
			method:    http.MethodPut,
			path:      "/simples/id/fresh",
			body:      simpleBody("id"),
			expectOp:  admission.Create,
			expectSub: "fresh",
			expectEqs: turboEqs,
			// The fixture validates with the update attributes; only mutation sees CREATE.
			mutateOnly: true,
		},
		{
			name:        "patch",
			method:      http.MethodPatch,
			path:        "/simples/id/turbo",
			body:        []byte(`{"other":"baz"}`),
			contentType: "application/merge-patch+json",
			expectOp:    admission.Update,
			expectSub:   "turbo",
			expectEqs:   turboEqs,
		},
		{name: "delete", method: http.MethodDelete, path: "/simples/id", expectOp: admission.Delete, expectEqs: baseEqs},
		{name: "delete collection", method: http.MethodDelete, path: "/simples", expectOp: admission.Delete, expectEqs: baseEqs},
		{name: "connect", method: http.MethodPost, path: "/simples/id/connect", expectOp: admission.Connect, expectSub: "connect", expectEqs: connectEqs},
		{name: "non-declaring endpoint", method: http.MethodPut, path: "/others/id", body: simpleBody("id"), expectOp: admission.Update},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			recorder.reset()
			req, err := http.NewRequestWithContext(t.Context(), tc.method, base+tc.path, bytes.NewReader(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			if tc.contentType != "" {
				req.Header.Set("Content-Type", tc.contentType)
			} else if tc.body != nil {
				req.Header.Set("Content-Type", "application/json")
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			respBody, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode >= 300 {
				t.Fatalf("unexpected status %d: %s", resp.StatusCode, respBody)
			}

			calls := recorder.reset()
			phases := map[string]bool{}
			for _, c := range calls {
				if c.operation != tc.expectOp || c.subresource != tc.expectSub {
					continue
				}
				phases[c.phase] = true
				if !reflect.DeepEqual(c.eqs, tc.expectEqs) {
					t.Errorf("%s: expected admission equivalents %#v, got %#v", c.phase, tc.expectEqs, c.eqs)
				}
			}
			if !phases["mutate"] || (!tc.mutateOnly && !phases["validate"]) {
				t.Fatalf("expected mutating and validating admission calls for %s %q, got %#v", tc.expectOp, tc.expectSub, calls)
			}
		})
	}
}

func TestInstallRejectsInvalidAdmissionEquivalents(t *testing.T) {
	group := APIGroupVersion{
		Storage: map[string]rest.Storage{
			"simples": &SimpleRESTStorage{},
			"simples/turbo": &equivalentsStorage{
				SimpleRESTStorageWithDeleteCollection: &SimpleRESTStorageWithDeleteCollection{},
				eqs:                                   []admission.Equivalent{{Subresource: "turbo"}},
			},
		},
		Root:                       "/" + prefix,
		GroupVersion:               testGroupVersion,
		OptionsExternalVersion:     &testGroupVersion,
		Serializer:                 codecs,
		Creater:                    scheme,
		Convertor:                  scheme,
		TypeConverter:              managedfields.NewDeducedTypeConverter(),
		UnsafeConvertor:            runtime.UnsafeObjectConvertor(scheme),
		Defaulter:                  scheme,
		Typer:                      scheme,
		Namer:                      namer,
		EquivalentResourceRegistry: runtime.NewEquivalentResourceRegistry(),
		ParameterCodec:             parameterCodec,
	}
	_, _, err := group.InstallREST(restful.NewContainer())
	const expectErr = `error in registering resource: simples/turbo, invalid admission equivalents: [0]: an endpoint may not declare itself (subresource "turbo") as an admission equivalent`
	if err == nil || err.Error() != expectErr {
		t.Fatalf("expected error %q, got %v", expectErr, err)
	}
}

func TestValidateAdmissionEquivalents(t *testing.T) {
	testcases := []struct {
		name        string
		subresource string
		eqs         []admission.Equivalent
		expectErrs  []string
	}{{
		name: "empty",
	}, {
		name:        "valid",
		subresource: "turbo",
		eqs: []admission.Equivalent{
			{Subresource: "", Operations: []admission.Operation{admission.Create, admission.Update}},
			{Subresource: "resize"},
			{Subresource: "extras", Operations: []admission.Operation{admission.Delete, admission.Connect}},
		},
	}, {
		name:        "self reference on subresource",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "turbo"}},
		expectErrs:  []string{`may not declare itself (subresource "turbo")`},
	}, {
		name:        "self reference on resource",
		subresource: "",
		eqs:         []admission.Equivalent{{Subresource: ""}},
		expectErrs:  []string{`may not declare itself (subresource "")`},
	}, {
		name:        "invalid operation",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "", Operations: []admission.Operation{"PATCH"}}},
		expectErrs:  []string{`invalid operation "PATCH"`},
	}, {
		name:        "wildcard operation",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "", Operations: []admission.Operation{"*"}}},
		expectErrs:  []string{`invalid operation "*"`},
	}, {
		name:        "slash",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "resize/x"}},
		expectErrs:  []string{`invalid subresource "resize/x"`},
	}, {
		name:        "wildcard subresource",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "*"}},
		expectErrs:  []string{`invalid subresource "*"`},
	}, {
		name:        "all errors reported",
		subresource: "turbo",
		eqs:         []admission.Equivalent{{Subresource: "turbo"}, {Subresource: "*", Operations: []admission.Operation{"GET"}}},
		expectErrs:  []string{`[0]: an endpoint may not declare itself`, `[1]: invalid subresource "*"`, `[1]: invalid operation "GET"`},
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAdmissionEquivalents(tc.subresource, tc.eqs)
			if len(tc.expectErrs) == 0 {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected errors %q, got none", tc.expectErrs)
			}
			for _, e := range tc.expectErrs {
				if !strings.Contains(err.Error(), e) {
					t.Errorf("expected error containing %q, got %q", e, err.Error())
				}
			}
		})
	}
}
