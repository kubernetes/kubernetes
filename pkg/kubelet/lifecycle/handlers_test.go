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

package lifecycle

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestResolvePortInt(t *testing.T) {
	expected := 80
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrInt, IntVal: expected}, &api.Container{})
	if port != expected {
		t.Errorf("expected: %d, saw: %d", expected, port)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolvePortString(t *testing.T) {
	expected := 80
	name := "foo"
	container := &api.Container{
		Ports: []api.ContainerPort{
			{Name: name, ContainerPort: expected},
		},
	}
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrString, StrVal: name}, container)
	if port != expected {
		t.Errorf("expected: %d, saw: %d", expected, port)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolvePortStringUnknown(t *testing.T) {
	expected := 80
	name := "foo"
	container := &api.Container{
		Ports: []api.ContainerPort{
			{Name: "bar", ContainerPort: expected},
		},
	}
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrString, StrVal: name}, container)
	if port != -1 {
		t.Errorf("expected: -1, saw: %d", port)
	}
	if err == nil {
		t.Error("unexpected non-error")
	}
}
