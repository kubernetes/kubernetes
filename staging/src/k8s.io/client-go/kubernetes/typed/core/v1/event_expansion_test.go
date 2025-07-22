/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"io"
	"net/http"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	restfake "k8s.io/client-go/rest/fake"
)

func TestCreateWithEventNamespace(t *testing.T) {
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
	}
	cli := &restfake.RESTClient{
		Client: restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}
			return resp, nil
		}),
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
	}

	tests := []struct {
		name    string
		ns      string
		event   *v1.Event
		wantErr bool
	}{
		{
			name:  "create event",
			ns:    "default",
			event: event,
		},
		{
			name:    "create event with different namespace",
			ns:      "other",
			event:   event,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newEvents(New(cli), tt.ns)
			_, err := e.CreateWithEventNamespace(tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("CreateWithEventNamespace() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestPatchWithEventNamespace(t *testing.T) {
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
	}
	cli := &restfake.RESTClient{
		Client: restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}
			return resp, nil
		}),
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
	}

	tests := []struct {
		name    string
		ns      string
		event   *v1.Event
		data    []byte
		wantErr bool
	}{
		{
			name:  "patch event",
			ns:    "default",
			event: event,
			data:  []byte{},
		},
		{
			name:    "patch event with different namespace",
			ns:      "other",
			event:   event,
			data:    []byte{},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newEvents(New(cli), tt.ns)
			_, err := e.PatchWithEventNamespace(tt.event, tt.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("PatchWithEventNamespace() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestUpdateWithEventNamespace(t *testing.T) {
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
	}
	cli := &restfake.RESTClient{
		Client: restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}
			return resp, nil
		}),
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
	}

	tests := []struct {
		name    string
		ns      string
		event   *v1.Event
		wantErr bool
	}{
		{
			name:  "patch event",
			ns:    "default",
			event: event,
		},
		{
			name:    "patch event with different namespace",
			ns:      "other",
			event:   event,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newEvents(New(cli), tt.ns)
			_, err := e.UpdateWithEventNamespace(tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("UpdateWithEventNamespace() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}
