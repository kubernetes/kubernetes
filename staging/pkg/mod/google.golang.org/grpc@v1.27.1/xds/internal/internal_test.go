/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package internal

import (
	"reflect"
	"strings"
	"testing"
	"unicode"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	"github.com/google/go-cmp/cmp"
)

const ignorePrefix = "XXX_"

func ignore(name string) bool {
	if !unicode.IsUpper([]rune(name)[0]) {
		return true
	}
	return strings.HasPrefix(name, ignorePrefix)
}

// A reflection based test to make sure internal.Locality contains all the
// fields (expect for XXX_) from the proto message.
func TestLocalityMatchProtoMessage(t *testing.T) {
	want1 := make(map[string]string)
	for ty, i := reflect.TypeOf(Locality{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want1[f.Name] = f.Type.Name()
	}

	want2 := make(map[string]string)
	for ty, i := reflect.TypeOf(corepb.Locality{}), 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		if ignore(f.Name) {
			continue
		}
		want2[f.Name] = f.Type.Name()
	}

	if !reflect.DeepEqual(want1, want2) {
		t.Fatalf("internal type and proto message have different fields:\n%+v", cmp.Diff(want1, want2))
	}
}
