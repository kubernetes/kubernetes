/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package scaler

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/restclient"
)

func getJSON(version, kind, name string, replicas int, selector string) []byte {
	return []byte(fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "metadata": {"name": %q}, "status": {"replicas": %f, "selector": %s}}`, version, kind, name, float64(replicas), selector))
}

func getClientServer(gv *unversioned.GroupVersion, h func(http.ResponseWriter, *http.Request)) (*Client, *httptest.Server, error) {
	srv := httptest.NewServer(http.HandlerFunc(h))
	cl, err := NewClient(&restclient.Config{
		Host: srv.URL,
	})
	if err != nil {
		srv.Close()
		return nil, nil, err
	}
	return cl, srv, nil
}

func TestGetReplicasAndSelector(t *testing.T) {
	replicas := 5
	sel := &unversioned.LabelSelector{
		MatchLabels: map[string]string{"foo": "bar"},
		MatchExpressions: []unversioned.LabelSelectorRequirement{
			{
				Key:      "lkfoo",
				Operator: unversioned.LabelSelectorOpIn,
				Values:   []string{"lbar", "lbaz"},
			},
			{
				Key:      "lkexist",
				Operator: unversioned.LabelSelectorOpExists,
			},
		},
	}
	labelSelector, err := unversioned.LabelSelectorAsSelector(sel)
	if err != nil {
		t.Errorf("failed to convert label selector to selector: %v", err)
	}

	tcs := []struct {
		namespace string
		name      string
		path      string
		scaleRef  extensions.SubresourceReference
		resp      []byte
		replicas  int
		selector  string
	}{
		{
			namespace: "testns",
			name:      "extrc",
			path:      "/apis/extensions/v1beta1/namespaces/testns/replicationcontrollers/extrc/scale",
			scaleRef: extensions.SubresourceReference{
				Kind:        "ReplicationController",
				Name:        "extrc",
				APIVersion:  "extensions/v1beta1",
				Subresource: "scale",
			},
			resp:     getJSON("extensions/v1beta1", "Scale", "extrc", replicas, `{"foo": "bar"}`),
			replicas: replicas,
			selector: `foo=bar`,
		},
		{
			namespace: "testns",
			name:      "extdep",
			path:      "/apis/extensions/v1beta1/namespaces/testns/deployments/extdep/scale",
			scaleRef: extensions.SubresourceReference{
				Kind:        "Deployment",
				Name:        "extdep",
				APIVersion:  "extensions/v1beta1",
				Subresource: "scale",
			},
			resp:     getJSON("autoscaling/v1", "Scale", "extdep", replicas, fmt.Sprintf("%q", labelSelector.String())),
			replicas: replicas,
			selector: labelSelector.String(),
		},
	}
	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "extensions", Version: "v1beta1"}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "GET" {
				t.Errorf("Get(%q) got HTTP method %s. wanted GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Get(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			w.Write(tc.resp)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		got, err := cl.Get(tc.scaleRef, tc.namespace)
		if err != nil {
			t.Errorf("unexpected error when getting %q: %v", tc.name, err)
			continue
		}

		if r, err := got.Replicas(); err != nil {
			t.Errorf("Get(%q) unexpected replicas error: %v", tc.name, err)
		} else if r != tc.replicas {
			t.Errorf("Get(%q) want replicas: %d\tgot: %d", tc.name, tc.replicas, r)
		}
		if s, err := got.Selector(); err != nil {
			t.Errorf("Get(%q) unexpected selector error: %v", tc.name, err)
		} else if s.String() != tc.selector {
			t.Errorf("Get(%q) want selector: %s\tgot: %s", tc.name, tc.selector, s)
		}
	}
}

func TestSetReplicas(t *testing.T) {
	initReplicas := 5
	finalReplicas := 10
	sel := &unversioned.LabelSelector{
		MatchLabels: map[string]string{"foo": "bar"},
		MatchExpressions: []unversioned.LabelSelectorRequirement{
			{
				Key:      "lkfoo",
				Operator: unversioned.LabelSelectorOpIn,
				Values:   []string{"lbar", "lbaz"},
			},
			{
				Key:      "lkexist",
				Operator: unversioned.LabelSelectorOpExists,
			},
		},
	}
	labelSelector, err := unversioned.LabelSelectorAsSelector(sel)
	if err != nil {
		t.Errorf("failed to convert label selector to selector: %v", err)
	}

	tcs := []struct {
		namespace string
		name      string
		path      string
		scaleRef  extensions.SubresourceReference
		resp      []byte
		replicas  int
		selector  string
	}{
		{
			namespace: "testns",
			name:      "extrc",
			path:      "/apis/extensions/v1beta1/namespaces/testns/replicationcontrollers/extrc/scale",
			scaleRef: extensions.SubresourceReference{
				Kind:        "ReplicationController",
				Name:        "extrc",
				APIVersion:  "extensions/v1beta1",
				Subresource: "scale",
			},
			resp:     getJSON("extensions/v1beta1", "Scale", "extrc", initReplicas, `{"foo": "bar"}`),
			replicas: finalReplicas,
		},
		{
			namespace: "testns",
			name:      "extdep",
			path:      "/apis/extensions/v1beta1/namespaces/testns/deployments/extdep/scale",
			scaleRef: extensions.SubresourceReference{
				Kind:        "Deployment",
				Name:        "extdep",
				APIVersion:  "extensions/v1beta1",
				Subresource: "scale",
			},
			resp:     getJSON("autoscaling/v1", "Scale", "extdep", initReplicas, fmt.Sprintf("%q", labelSelector.String())),
			replicas: finalReplicas,
		},
	}

	for _, tc := range tcs {
		gv := &unversioned.GroupVersion{Group: "extensions", Version: "v1beta1"}
		cl, srv, err := getClientServer(gv, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "PUT" && r.Method != "GET" {
				t.Errorf("Update(%q) got HTTP method %s. wanted PUT or GET", tc.name, r.Method)
			}

			if r.URL.Path != tc.path {
				t.Errorf("Update(%q) got path %s. wanted %s", tc.name, r.URL.Path, tc.path)
			}

			if r.Method == "GET" {
				w.Write(tc.resp)
				return
			}

			data, err := ioutil.ReadAll(r.Body)
			if err != nil {
				t.Errorf("Update(%q) unexpected error reading body: %v", tc.name, err)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			w.Write(data)
		})
		if err != nil {
			t.Errorf("unexpected error when creating client: %v", err)
			continue
		}
		defer srv.Close()

		obj, err := cl.Get(tc.scaleRef, tc.namespace)
		if err != nil {
			t.Errorf("unexpected error when getting %q: %v", tc.name, err)
			continue
		}

		obj.SetReplicas(tc.replicas)
		_, err = cl.Update(tc.scaleRef, tc.namespace, obj)
		if err != nil {
			t.Errorf("unexpected error when updating %q: %v", tc.name, err)
			continue
		}
	}
}
