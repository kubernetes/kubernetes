/*
Copyright 2014 The Kubernetes Authors.

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

package json

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestSimpleMetaFactoryInterpret(t *testing.T) {
	factory := SimpleMetaFactory{}
	gvk, err := factory.Interpret([]byte(`{"apiVersion":"1","kind":"object"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gvk.Version != "1" || gvk.Kind != "object" {
		t.Errorf("unexpected interpret: %#v", gvk)
	}

	// no kind or version
	gvk, err = factory.Interpret([]byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gvk.Version != "" || gvk.Kind != "" {
		t.Errorf("unexpected interpret: %#v", gvk)
	}

	// unparsable
	_, err = factory.Interpret([]byte(`{`))
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSimpleMetaFactoryInterpretInvalidKindType(t *testing.T) {
	factory := SimpleMetaFactory{}
	_, err := factory.Interpret([]byte(`{"kind":true}`))
	if err == nil {
		t.Fatalf("expected an error for invalid kind type, got nil")
	}
	// The underlying json error must be recoverable via errors.As so that
	// callers can classify it as a client (4xx) error instead of a 500.
	var typeErr *json.UnmarshalTypeError
	if !errors.As(err, &typeErr) {
		t.Errorf("expected wrapped *json.UnmarshalTypeError, got %T: %v", err, err)
	}
}
