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
	"strings"

	"sigs.k8s.io/randfill"

	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/audit"
)

// Funcs returns the fuzzer functions for the audit api group.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(e *audit.Event, c randfill.Continue) {
			c.FillNoCustom(e)
			switch c.Bool() {
			case true:
				e.RequestObject = nil
			case false:
				e.RequestObject = &runtime.Unknown{
					TypeMeta:    runtime.TypeMeta{APIVersion: "", Kind: ""},
					Raw:         []byte(`{"apiVersion":"","kind":"Pod","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			}
			switch c.Bool() {
			case true:
				e.ResponseObject = nil
			case false:
				e.ResponseObject = &runtime.Unknown{
					TypeMeta:    runtime.TypeMeta{APIVersion: "", Kind: ""},
					Raw:         []byte(`{"apiVersion":"","kind":"Pod","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			}
		},
		func(o *audit.ObjectReference, c randfill.Continue) {
			c.FillNoCustom(o)
			switch c.Intn(3) {
			case 0:
				// core api group
				o.APIGroup = ""
				o.APIVersion = "v1"
			case 1:
				// other group
				o.APIGroup = "rbac.authorization.k8s.io"
				o.APIVersion = "v1beta1"
			default:
				// use random value, but without / as it is used as separator
				o.APIGroup = strings.Replace(o.APIGroup, "/", "-", -1)
				o.APIVersion = strings.Replace(o.APIVersion, "/", "-", -1)
			}
		},
	}
}
