/*
Copyright 2017 The Kubernetes Authors.

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

package fuzzer

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	kjson "k8s.io/apimachinery/pkg/util/json"
)

// FuzzerFuncs returns a list of func(*SomeType, c fuzz.Continue) functions.
type FuzzerFuncs func(codecs runtimeserializer.CodecFactory) []interface{}

// FuzzerFor can randomly populate api objects that are destined for version.
func FuzzerFor(funcs FuzzerFuncs, src rand.Source, codecs runtimeserializer.CodecFactory) *fuzz.Fuzzer {
	f := fuzz.New().NilChance(.5).NumElements(0, 1)
	if src != nil {
		f.RandSource(src)
	}
	f.Funcs(funcs(codecs)...)
	return f
}

// MergeFuzzerFuncs will merge the given funcLists, overriding early funcs with later ones if there first
// argument has the same type.
func MergeFuzzerFuncs(funcs ...FuzzerFuncs) FuzzerFuncs {
	return FuzzerFuncs(func(codecs runtimeserializer.CodecFactory) []interface{} {
		result := []interface{}{}
		for _, f := range funcs {
			if f != nil {
				result = append(result, f(codecs)...)
			}
		}
		return result
	})
}

func NormalizeJSONRawExtension(ext *runtime.RawExtension) {
	if json.Valid(ext.Raw) {
		// RawExtension->JSON encodes struct fields in field index order while map[string]interface{}->JSON encodes
		// struct fields (i.e. keys in the map) lexicographically. We have to sort the fields here to ensure the
		// JSON in the (RawExtension->)JSON->map[string]interface{}->JSON round trip results in identical JSON.
		var u any
		err := kjson.Unmarshal(ext.Raw, &u)
		if err != nil {
			panic(fmt.Sprintf("Failed to encode object: %v", err))
		}
		ext.Raw, err = kjson.Marshal(&u)
		if err != nil {
			panic(fmt.Sprintf("Failed to encode object: %v", err))
		}
	}
}
