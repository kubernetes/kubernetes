/*
Copyright 2024 The Kubernetes Authors.

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

package modes_test

import (
	"errors"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"
)

var encModeNames = map[cbor.EncMode]string{
	modes.Encode:                 "Encode",
	modes.EncodeNondeterministic: "EncodeNondeterministic",
}

var allEncModes = []cbor.EncMode{
	modes.Encode,
	modes.EncodeNondeterministic,
}

var decModeNames = map[cbor.DecMode]string{
	modes.Decode:    "Decode",
	modes.DecodeLax: "DecodeLax",
}

var allDecModes = []cbor.DecMode{
	modes.Decode,
	modes.DecodeLax,
}

func assertNilError(t *testing.T, e error) {
	if e != nil {
		t.Errorf("expected nil error, got: %v", e)
	}
}

func assertOnConcreteError[E error](fn func(*testing.T, E)) func(t *testing.T, e error) {
	return func(t *testing.T, ei error) {
		var ec E
		if !errors.As(ei, &ec) {
			t.Errorf("expected concrete error type %T, got %T: %v", ec, ei, ei)
			return
		}
		fn(t, ec)
	}
}
