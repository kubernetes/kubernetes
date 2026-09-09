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

package apis

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration/install"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	apiregistrationv1beta1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
)

func TestRoundtripToUnstructured(t *testing.T) {
	scheme := runtime.NewScheme()
	install.Install(scheme)

	roundtrip.RoundtripToUnstructured(t, scheme, fuzzer.MergeFuzzerFuncs(), sets.New(
		apiregistrationv1.SchemeGroupVersion.WithKind("CreateOptions"),
		apiregistrationv1.SchemeGroupVersion.WithKind("PatchOptions"),
		apiregistrationv1.SchemeGroupVersion.WithKind("UpdateOptions"),
		apiregistrationv1beta1.SchemeGroupVersion.WithKind("CreateOptions"),
		apiregistrationv1beta1.SchemeGroupVersion.WithKind("PatchOptions"),
		apiregistrationv1beta1.SchemeGroupVersion.WithKind("UpdateOptions"),
	), nil)
}
