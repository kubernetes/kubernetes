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

package apiclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// dryRunRoundTripper answers clientset requests in-process by turning each one into
// the client-go testing Action the fake clients produce and running the reactor chain.
type dryRunRoundTripper struct {
	dryRun *DryRun
	// listKinds maps a resource to the kind the object tracker needs to build LIST results.
	listKinds map[schema.GroupVersionResource]schema.GroupVersionKind
}

func newDryRunRoundTripper(d *DryRun) *dryRunRoundTripper {
	rt := &dryRunRoundTripper{
		dryRun:    d,
		listKinds: map[schema.GroupVersionResource]schema.GroupVersionKind{},
	}
	for gvk := range clientsetscheme.Scheme.AllKnownTypes() {
		kind := strings.TrimSuffix(gvk.Kind, "List")
		if kind == "" || kind == gvk.Kind {
			continue
		}
		gvk.Kind = kind
		plural, _ := meta.UnsafeGuessKindToResource(gvk)
		rt.listKinds[plural] = gvk
	}
	return rt
}

// RoundTrip implements http.RoundTripper.
func (rt *dryRunRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		var err error
		body, err = io.ReadAll(req.Body)
		_ = req.Body.Close()
		if err != nil {
			return errorResponse(req, apierrors.NewBadRequest(err.Error())), nil
		}
	}

	if req.URL.Path == "/version" {
		data, err := json.Marshal(rt.dryRun.serverVersion)
		if err != nil {
			return errorResponse(req, err), nil
		}
		return jsonResponse(req, http.StatusOK, data), nil
	}

	action, err := rt.actionFor(req, body)
	if err != nil {
		return errorResponse(req, err), nil
	}
	obj, err := rt.dryRun.fake.Invokes(action, nil)
	if err != nil {
		return errorResponse(req, err), nil
	}
	if list, ok := action.(testing.ListAction); ok && obj != nil {
		if err := filterByLabels(obj, list.GetListRestrictions().Labels); err != nil {
			return errorResponse(req, err), nil
		}
	}
	if obj == nil {
		obj = &metav1.Status{Status: metav1.StatusSuccess, Code: http.StatusOK}
	}
	data, err := runtime.Encode(clientsetscheme.Codecs.LegacyCodec(action.GetResource().GroupVersion()), obj)
	if err != nil {
		return errorResponse(req, err), nil
	}
	return jsonResponse(req, http.StatusOK, data), nil
}

// actionFor maps a request to the testing Action the fake typed clients would have produced.
func (rt *dryRunRoundTripper) actionFor(req *http.Request, body []byte) (testing.Action, error) {
	gvr, namespace, name, subresource, ok := parseResourcePath(req.URL.Path)
	if !ok {
		return nil, &apierrors.StatusError{ErrStatus: metav1.Status{
			Status:  metav1.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  metav1.StatusReasonNotFound,
			Message: fmt.Sprintf("the dry-run client does not serve %s %s", req.Method, req.URL.Path),
		}}
	}
	query := req.URL.Query()

	switch {
	case req.Method == http.MethodGet && name != "":
		return testing.NewGetSubresourceAction(gvr, namespace, subresource, name), nil
	case req.Method == http.MethodGet && query.Get("watch") == "":
		opts := metav1.ListOptions{
			LabelSelector: query.Get("labelSelector"),
			FieldSelector: query.Get("fieldSelector"),
		}
		// NewListActionWithOptions panics on malformed selectors; reject them instead.
		if _, err := labels.Parse(opts.LabelSelector); err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		if _, err := fields.ParseSelector(opts.FieldSelector); err != nil {
			return nil, apierrors.NewBadRequest(err.Error())
		}
		return testing.NewListActionWithOptions(gvr, rt.listKinds[gvr], namespace, opts), nil
	case req.Method == http.MethodPost:
		obj, err := decodeBody(body)
		if err != nil {
			return nil, err
		}
		return testing.NewCreateSubresourceAction(gvr, name, subresource, namespace, obj), nil
	case req.Method == http.MethodPut:
		obj, err := decodeBody(body)
		if err != nil {
			return nil, err
		}
		return testing.NewUpdateSubresourceAction(gvr, subresource, namespace, obj), nil
	case req.Method == http.MethodPatch && name != "":
		patchType := types.PatchType(req.Header.Get("Content-Type"))
		return testing.NewPatchSubresourceAction(gvr, namespace, name, patchType, body, subresource), nil
	case req.Method == http.MethodDelete && name != "":
		return testing.NewDeleteSubresourceAction(gvr, subresource, namespace, name), nil
	}
	return nil, apierrors.NewMethodNotSupported(gvr.GroupResource(), req.Method+" "+req.URL.RequestURI())
}

// parseResourcePath splits an API request path into resource, namespace, name and subresource.
// ok is false for paths outside /api and /apis and for discovery paths without a resource.
func parseResourcePath(path string) (gvr schema.GroupVersionResource, namespace, name, subresource string, ok bool) {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	switch {
	case len(parts) >= 3 && parts[0] == "api":
		gvr.Version, parts = parts[1], parts[2:]
	case len(parts) >= 4 && parts[0] == "apis":
		gvr.Group, gvr.Version, parts = parts[1], parts[2], parts[3:]
	default:
		return gvr, "", "", "", false
	}
	if len(parts) > 2 && parts[0] == "namespaces" {
		namespace, parts = parts[1], parts[2:]
	}
	switch len(parts) {
	case 3:
		subresource = parts[2]
		fallthrough
	case 2:
		name = parts[1]
		fallthrough
	case 1:
		gvr.Resource = parts[0]
		return gvr, namespace, name, subresource, true
	}
	return gvr, "", "", "", false
}

// filterByLabels drops the list items sel does not match, as the typed fake clients did.
func filterByLabels(obj runtime.Object, sel labels.Selector) error {
	if sel == nil || sel.Empty() {
		return nil
	}
	items, err := meta.ExtractList(obj)
	if err != nil {
		return err
	}
	kept := items[:0]
	for _, item := range items {
		if m, err := meta.Accessor(item); err == nil && sel.Matches(labels.Set(m.GetLabels())) {
			kept = append(kept, item)
		}
	}
	return meta.SetList(obj, kept)
}

// decodeBody decodes a request body into the typed object the reactors expect.
func decodeBody(body []byte) (runtime.Object, error) {
	obj, _, err := clientsetscheme.Codecs.UniversalDeserializer().Decode(body, nil, nil)
	if err != nil {
		return nil, apierrors.NewBadRequest(err.Error())
	}
	return obj, nil
}

// errorResponse encodes err as a metav1.Status like the API server does, so that helpers
// such as apierrors.IsNotFound keep working for callers of the dry-run client.
func errorResponse(req *http.Request, err error) *http.Response {
	var apiStatus apierrors.APIStatus
	if !errors.As(err, &apiStatus) {
		apiStatus = apierrors.NewInternalError(err)
	}
	status := apiStatus.Status()
	if status.Code == 0 {
		status.Code = http.StatusInternalServerError
	}
	data, err := runtime.Encode(clientsetscheme.Codecs.LegacyCodec(metav1.SchemeGroupVersion), &status)
	if err != nil {
		return jsonResponse(req, http.StatusInternalServerError, []byte(err.Error()))
	}
	return jsonResponse(req, int(status.Code), data)
}

func jsonResponse(req *http.Request, code int, data []byte) *http.Response {
	return &http.Response{
		StatusCode:    code,
		Header:        http.Header{"Content-Type": []string{runtime.ContentTypeJSON}},
		Body:          io.NopCloser(bytes.NewReader(data)),
		ContentLength: int64(len(data)),
		Request:       req,
	}
}
