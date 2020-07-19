/*
Copyright 2019 The Kubernetes Authors.

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

package metadata

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/rest"
)

func TestClient(t *testing.T) {
	gvr := schema.GroupVersionResource{Group: "group", Version: "v1", Resource: "resource"}

	writeJSON := func(t *testing.T, w http.ResponseWriter, obj runtime.Object) {
		data, err := json.Marshal(obj)
		if err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}
	}

	testCases := []struct {
		name    string
		handler func(t *testing.T, w http.ResponseWriter, req *http.Request)
		want    func(t *testing.T, client *Client)
	}{
		{
			name: "GET is able to convert a JSON object to PartialObjectMetadata",
			handler: func(t *testing.T, w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Accept") != "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json" {
					t.Fatal(req.Header.Get("Accept"))
				}
				if req.Method != "GET" && req.URL.String() != "/apis/group/v1/namespaces/ns/resource/name" {
					t.Fatal(req.URL.String())
				}
				writeJSON(t, w, &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "name",
						Namespace: "ns",
					},
				})
			},
			want: func(t *testing.T, client *Client) {
				obj, err := client.Resource(gvr).Namespace("ns").Get(context.TODO(), "name", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				expect := &metav1.PartialObjectMetadata{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "name",
						Namespace: "ns",
					},
				}
				if !reflect.DeepEqual(expect, obj) {
					t.Fatal(diff.ObjectReflectDiff(expect, obj))
				}
			},
		},

		{
			name: "LIST is able to convert a JSON object to PartialObjectMetadata",
			handler: func(t *testing.T, w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Accept") != "application/vnd.kubernetes.protobuf;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1,application/json" {
					t.Fatal(req.Header.Get("Accept"))
				}
				if req.Method != "GET" && req.URL.String() != "/apis/group/v1/namespaces/ns/resource" {
					t.Fatal(req.URL.String())
				}
				writeJSON(t, w, &corev1.PodList{
					TypeMeta: metav1.TypeMeta{
						Kind:       "PodList",
						APIVersion: "v1",
					},
					ListMeta: metav1.ListMeta{
						ResourceVersion: "253",
					},
					Items: []corev1.Pod{
						{
							TypeMeta: metav1.TypeMeta{
								Kind:       "Pod",
								APIVersion: "v1",
							},
							ObjectMeta: metav1.ObjectMeta{
								Name:      "name",
								Namespace: "ns",
							},
						},
					},
				})
			},
			want: func(t *testing.T, client *Client) {
				objs, err := client.Resource(gvr).Namespace("ns").List(context.TODO(), metav1.ListOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if objs.GetResourceVersion() != "253" {
					t.Fatal(objs)
				}
				expect := []metav1.PartialObjectMetadata{
					{
						TypeMeta: metav1.TypeMeta{
							Kind:       "Pod",
							APIVersion: "v1",
						},
						ObjectMeta: metav1.ObjectMeta{
							Name:      "name",
							Namespace: "ns",
						},
					},
				}
				if !reflect.DeepEqual(expect, objs.Items) {
					t.Fatal(diff.ObjectReflectDiff(expect, objs.Items))
				}
			},
		},

		{
			name: "GET fails if the object is JSON and has no kind",
			handler: func(t *testing.T, w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Accept") != "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json" {
					t.Fatal(req.Header.Get("Accept"))
				}
				if req.Method != "GET" && req.URL.String() != "/apis/group/v1/namespaces/ns/resource/name" {
					t.Fatal(req.URL.String())
				}
				writeJSON(t, w, &corev1.Pod{
					TypeMeta: metav1.TypeMeta{},
					ObjectMeta: metav1.ObjectMeta{
						UID: "123",
					},
				})
			},
			want: func(t *testing.T, client *Client) {
				obj, err := client.Resource(gvr).Namespace("ns").Get(context.TODO(), "name", metav1.GetOptions{})
				if err == nil || !runtime.IsMissingKind(err) {
					t.Fatal(err)
				}
				if obj != nil {
					t.Fatal(obj)
				}
			},
		},

		{
			name: "GET fails if the object is JSON and has no apiVersion",
			handler: func(t *testing.T, w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Accept") != "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json" {
					t.Fatal(req.Header.Get("Accept"))
				}
				if req.Method != "GET" && req.URL.String() != "/apis/group/v1/namespaces/ns/resource/name" {
					t.Fatal(req.URL.String())
				}
				writeJSON(t, w, &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind: "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						UID: "123",
					},
				})
			},
			want: func(t *testing.T, client *Client) {
				obj, err := client.Resource(gvr).Namespace("ns").Get(context.TODO(), "name", metav1.GetOptions{})
				if err == nil || !runtime.IsMissingVersion(err) {
					t.Fatal(err)
				}
				if obj != nil {
					t.Fatal(obj)
				}
			},
		},

		{
			name: "GET fails if the object is JSON and not clearly metadata",
			handler: func(t *testing.T, w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Accept") != "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json" {
					t.Fatal(req.Header.Get("Accept"))
				}
				if req.Method != "GET" && req.URL.String() != "/apis/group/v1/namespaces/ns/resource/name" {
					t.Fatal(req.URL.String())
				}
				writeJSON(t, w, &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{},
				})
			},
			want: func(t *testing.T, client *Client) {
				obj, err := client.Resource(gvr).Namespace("ns").Get(context.TODO(), "name", metav1.GetOptions{})
				if err == nil || !strings.Contains(err.Error(), "object does not appear to match the ObjectMeta schema") {
					t.Fatal(err)
				}
				if obj != nil {
					t.Fatal(obj)
				}
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) { tt.handler(t, w, req) }))
			defer s.Close()

			cfg := ConfigFor(&rest.Config{Host: s.URL})
			client := NewForConfigOrDie(cfg).(*Client)
			tt.want(t, client)
		})
	}
}
