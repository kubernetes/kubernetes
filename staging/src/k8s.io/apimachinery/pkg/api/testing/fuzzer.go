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

package testing

import (
	"math/rand"
	"reflect"
	"strconv"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
)

func GenericFuzzerFuncs(t TestingCommon, codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(q *resource.Quantity, c fuzz.Continue) {
			*q = *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
		},
		func(j *int, c fuzz.Continue) {
			*j = int(c.Int31())
		},
		func(j **int, c fuzz.Continue) {
			if c.RandBool() {
				i := int(c.Int31())
				*j = &i
			} else {
				*j = nil
			}
		},
		func(j *runtime.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *metav1.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *metav1.ObjectMeta, c fuzz.Continue) {
			j.Name = c.RandString()
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
			j.UID = types.UID(c.RandString())
			j.GenerateName = c.RandString()

			var sec, nsec int64
			c.Fuzz(&sec)
			c.Fuzz(&nsec)
			j.CreationTimestamp = metav1.Unix(sec, nsec).Rfc3339Copy()
		},
		func(j *metav1.ListMeta, c fuzz.Continue) {
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
		},
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					// We do not set TypeMeta here because it is not carried through a round trip
					Raw:         []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			} else {
				types := []runtime.Object{&metav1.Status{}, &metav1.APIGroup{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// Pick an arbitrary type and fuzz it
			types := []runtime.Object{&metav1.Status{}, &metav1.APIGroup{}}
			obj := types[c.Rand.Intn(len(types))]
			c.Fuzz(obj)

			// Find a codec for converting the object to raw bytes.  This is necessary for the
			// api version and kind to be correctly set be serialization.
			var codec = TestCodec(codecs, metav1.SchemeGroupVersion)

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				t.Errorf("Failed to encode object: %v", err)
				return
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

// TestingCommon abstracts testing.T and testing.B
type TestingCommon interface {
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
}

var (
	_ TestingCommon = &testing.T{}
	_ TestingCommon = &testing.B{}
)

// FuzzerFor can randomly populate api objects that are destined for version.
func FuzzerFor(funcs []interface{}, src rand.Source) *fuzz.Fuzzer {
	f := fuzz.New().NilChance(.5).NumElements(0, 1)
	if src != nil {
		f.RandSource(src)
	}
	f.Funcs(funcs...)
	return f
}

// MergeFuzzerFuncs will merge the given funcLists, overriding early funcs with later ones if there first
// argument has the same type.
func MergeFuzzerFuncs(t TestingCommon, funcLists ...[]interface{}) []interface{} {
	funcMap := map[string]interface{}{}
	for _, list := range funcLists {
		for _, f := range list {
			fT := reflect.TypeOf(f)
			if fT.Kind() != reflect.Func || fT.NumIn() != 2 {
				t.Errorf("Fuzzer func with invalid type: %v", fT)
				continue
			}
			funcMap[fT.In(0).String()] = f
		}
	}

	result := []interface{}{}
	for _, f := range funcMap {
		result = append(result, f)
	}
	return result
}

func DefaultFuzzers(t TestingCommon, codecFactory runtimeserializer.CodecFactory, fuzzerFuncs []interface{}) *fuzz.Fuzzer {
	seed := rand.Int63()

	return FuzzerFor(
		MergeFuzzerFuncs(t,
			GenericFuzzerFuncs(t, codecFactory),
			fuzzerFuncs,
		),
		rand.NewSource(seed),
	)

}
