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
	"fmt"
	"strconv"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
)

func genericFuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
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
			var codec = apitesting.TestCodec(codecs, metav1.SchemeGroupVersion)

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				panic(fmt.Sprintf("Failed to encode object: %v", err))
			}

			// strip trailing newlines which do not survive roundtrips
			for len(bytes) >= 1 && bytes[len(bytes)-1] == 10 {
				bytes = bytes[:len(bytes)-1]
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

func v1FuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
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
	}
}

func v1alpha1FuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(r *metav1alpha1.TableRow, c fuzz.Continue) {
			c.Fuzz(&r.Object)
			c.Fuzz(&r.Conditions)
			if len(r.Conditions) == 0 {
				r.Conditions = nil
			}
			n := c.Intn(10)
			if n > 0 {
				r.Cells = make([]interface{}, n)
			}
			for i := range r.Cells {
				t := c.Intn(6)
				switch t {
				case 0:
					r.Cells[i] = c.RandString()
				case 1:
					r.Cells[i] = c.Uint64()
				case 2:
					r.Cells[i] = c.RandBool()
				case 3:
					// maps roundtrip as map[interface{}]interface{}, but the json codec cannot encode that
					// TODO: get maps to roundtrip properly
					/*
						x := map[string]interface{}{}
						for j := c.Intn(10) + 1; j >= 0; j-- {
							x[c.RandString()] = c.RandString()
						}
						r.Cells[i] = x
					*/
				case 4:
					x := make([]interface{}, c.Intn(10))
					for i := range x {
						x[i] = c.Uint64()
					}
					r.Cells[i] = x
				default:
					r.Cells[i] = nil
				}
			}
		},
	}
}

var Funcs = fuzzer.MergeFuzzerFuncs(
	genericFuzzerFuncs,
	v1FuzzerFuncs,
	v1alpha1FuzzerFuncs,
)
