/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"

	fuzz "github.com/google/gofuzz"

	"k8s.io/api/core/v1"
	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	genericfuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	admissionregistrationfuzzer "k8s.io/kubernetes/pkg/apis/admissionregistration/fuzzer"
	appsfuzzer "k8s.io/kubernetes/pkg/apis/apps/fuzzer"
	autoscalingfuzzer "k8s.io/kubernetes/pkg/apis/autoscaling/fuzzer"
	batchfuzzer "k8s.io/kubernetes/pkg/apis/batch/fuzzer"
	certificatesfuzzer "k8s.io/kubernetes/pkg/apis/certificates/fuzzer"
	api "k8s.io/kubernetes/pkg/apis/core"
	corefuzzer "k8s.io/kubernetes/pkg/apis/core/fuzzer"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsfuzzer "k8s.io/kubernetes/pkg/apis/extensions/fuzzer"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	networkingfuzzer "k8s.io/kubernetes/pkg/apis/networking/fuzzer"
	policyfuzzer "k8s.io/kubernetes/pkg/apis/policy/fuzzer"
	rbacfuzzer "k8s.io/kubernetes/pkg/apis/rbac/fuzzer"
	storagefuzzer "k8s.io/kubernetes/pkg/apis/storage/fuzzer"
)

// overrideGenericFuncs override some generic fuzzer funcs from k8s.io/apiserver in order to have more realistic
// values in a Kubernetes context.
func overrideGenericFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					// We do not set TypeMeta here because it is not carried through a round trip
					Raw:         []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			} else {
				types := []runtime.Object{&api.Pod{}, &api.ReplicationController{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// Pick an arbitrary type and fuzz it
			types := []runtime.Object{&api.Pod{}, &extensions.Deployment{}, &api.Service{}}
			obj := types[c.Rand.Intn(len(types))]
			c.Fuzz(obj)

			var codec runtime.Codec
			switch obj.(type) {
			case *extensions.Deployment:
				codec = apitesting.TestCodec(codecs, extensionsv1beta1.SchemeGroupVersion)
			default:
				codec = apitesting.TestCodec(codecs, v1.SchemeGroupVersion)
			}

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				panic(fmt.Sprintf("Failed to encode object: %v", err))
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

// FuzzerFuncs is a list of fuzzer functions
var FuzzerFuncs = fuzzer.MergeFuzzerFuncs(
	genericfuzzer.Funcs,
	overrideGenericFuncs,
	corefuzzer.Funcs,
	extensionsfuzzer.Funcs,
	appsfuzzer.Funcs,
	batchfuzzer.Funcs,
	autoscalingfuzzer.Funcs,
	rbacfuzzer.Funcs,
	policyfuzzer.Funcs,
	certificatesfuzzer.Funcs,
	admissionregistrationfuzzer.Funcs,
	storagefuzzer.Funcs,
	networkingfuzzer.Funcs,
)
