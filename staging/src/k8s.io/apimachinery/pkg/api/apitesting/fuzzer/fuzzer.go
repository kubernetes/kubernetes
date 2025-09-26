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
	"reflect"

	"sigs.k8s.io/randfill"

	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	kjson "k8s.io/apimachinery/pkg/util/json"
)

// FuzzerFuncs returns a list of func(*SomeType, c randfill.Continue) functions.
type FuzzerFuncs func(codecs runtimeserializer.CodecFactory) []interface{}

// FuzzerFor can randomly populate api objects that are destined for version.
func FuzzerFor(funcs FuzzerFuncs, src rand.Source, codecs runtimeserializer.CodecFactory) *randfill.Filler {
	f := randfill.New().NilChance(.5).NumElements(0, 1)
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

// SchemeDefaultingFuzzerFuncs returns fuzzer functions that automatically apply
// scheme defaults to objects after fuzzing. This eliminates the need to manually
// duplicate defaulting logic in fuzzer functions.
//
// This function addresses the issue described in kubernetes/kubernetes#130791 where
// fuzzer functions were manually duplicating defaulting logic that was already
// registered in the scheme. Instead of manually setting defaults in each fuzzer
// function, this automatically applies the scheme's registered defaults.
//
// Usage:
//
//	// OLD WAY - Manual defaulting duplication
//	func(obj *SomeType, c randfill.Continue) {
//		c.FillNoCustom(obj)
//		// Manual defaulting duplication
//		if obj.FailurePolicy == nil {
//			p := SomeFailurePolicyType("Fail")
//			obj.FailurePolicy = &p
//		}
//		// ... more manual defaulting
//	}
//
//	// NEW WAY - Automatic scheme defaulting
//	fuzzer := fuzzer.FuzzerFor(
//		fuzzer.MergeFuzzerFuncs(
//			metafuzzer.Funcs,
//			fuzzer.SchemeDefaultingFuzzerFuncs(scheme), // Automatic defaults
//			customFuzzerFuncs, // No manual defaulting needed
//		),
//		rand.NewSource(seed),
//		codecs,
//	)
//
//	// In roundtrip tests:
//	roundtrip.RoundTripTestForScheme(t, scheme, fuzzer.MergeFuzzerFuncs(
//		metafuzzer.Funcs,
//		fuzzer.SchemeDefaultingFuzzerFuncs(scheme),
//		customFuzzerFuncs,
//	))
func SchemeDefaultingFuzzerFuncs(scheme *runtime.Scheme) FuzzerFuncs {
	return func(codecs runtimeserializer.CodecFactory) []interface{} {
		// Get all known types from the scheme
		knownTypes := scheme.AllKnownTypes()
		fuzzerFuncs := make([]interface{}, 0, len(knownTypes))

		for gvk, objType := range knownTypes {
			// Skip internal types as they don't have external defaults
			if gvk.Version == runtime.APIVersionInternal {
				continue
			}

			// Create a fuzzer function for this type that applies defaults
			fuzzerFunc := func(obj interface{}, c randfill.Continue) {
				// First, let the fuzzer fill the object normally
				c.FillNoCustom(obj)

				// Then apply scheme defaults
				if runtimeObj, ok := obj.(runtime.Object); ok {
					scheme.Default(runtimeObj)
				}
			}

			// Create a typed version of the function for this specific type
			typedFunc := reflect.MakeFunc(
				reflect.FuncOf(
					[]reflect.Type{reflect.PtrTo(objType), reflect.TypeOf((*randfill.Continue)(nil)).Elem()},
					[]reflect.Type{},
					false,
				),
				func(args []reflect.Value) []reflect.Value {
					fuzzerFunc(args[0].Interface(), args[1].Interface().(randfill.Continue))
					return nil
				},
			).Interface()

			fuzzerFuncs = append(fuzzerFuncs, typedFunc)
		}

		return fuzzerFuncs
	}
}
