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
	"sigs.k8s.io/randfill"

	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

// Funcs returns the fuzzer functions for the authorization api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		// Sort field names consistently in the RawExtension JSON encoding
		func(obj *authorizationapi.AuthorizationConditionsReview, c randfill.Continue) {
			c.FillNoCustom(obj)
			if obj.Request != nil && obj.Request.AdmissionControlData != nil {
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.Object)
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.OldObject)
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.Options)
			}
		},
		// Sort field names consistently in the RawExtension JSON encoding
		func(obj *authorizationv1alpha1.AuthorizationConditionsReview, c randfill.Continue) {
			c.FillNoCustom(obj)
			if obj.Request != nil && obj.Request.AdmissionControlData != nil {
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.Object)
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.OldObject)
				fuzzer.NormalizeJSONRawExtension(&obj.Request.AdmissionControlData.Options)
			}
		},
	}
}
