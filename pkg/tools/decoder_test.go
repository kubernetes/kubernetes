/*
Copyright 2014 Google Inc. All rights reserved.

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

package tools

import (
	"encoding/json"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestDecoder(t *testing.T) {
	out, in := io.Pipe()
	encoder := json.NewEncoder(in)
	decoder := NewAPIEventDecoder(out)

	expect := &api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	go func() {
		err := encoder.Encode(api.WatchEvent{watch.Added, api.APIObject{expect}})
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
	}()

	done := make(chan struct{})
	go func() {
		action, got, err := decoder.Decode()
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		if e, a := watch.Added, action; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := expect, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
		close(done)
	}()
	select {
	case <-done:
		break
	case <-time.After(10 * time.Second):
		t.Error("Timeout")
	}

	done = make(chan struct{})

	go func() {
		_, _, err := decoder.Decode()
		if err == nil {
			t.Errorf("Unexpected nil error")
		}
		close(done)
	}()

	decoder.Close()

	select {
	case <-done:
		break
	case <-time.After(10 * time.Second):
		t.Error("Timeout")
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	out, in := io.Pipe()
	decoder := NewAPIEventDecoder(out)

	done := make(chan struct{})

	go func() {
		_, _, err := decoder.Decode()
		if err == nil {
			t.Errorf("Unexpected nil error")
		}
		close(done)
	}()

	in.Close()

	select {
	case <-done:
		break
	case <-time.After(10 * time.Second):
		t.Error("Timeout")
	}
}
