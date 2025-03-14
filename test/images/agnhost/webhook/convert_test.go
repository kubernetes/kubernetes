/*
Copyright 2019 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/randfill"

	v1 "k8s.io/api/admission/v1"
	"k8s.io/api/admission/v1beta1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	admissionfuzzer "k8s.io/kubernetes/pkg/apis/admission/fuzzer"
)

func TestConvertAdmissionRequestToV1(t *testing.T) {
	f := fuzzer.FuzzerFor(admissionfuzzer.Funcs, rand.NewSource(rand.Int63()), serializer.NewCodecFactory(runtime.NewScheme()))
	for i := 0; i < 100; i++ {
		t.Run(fmt.Sprintf("Run %d/100", i), func(t *testing.T) {
			orig := &v1beta1.AdmissionRequest{}
			f.Fill(orig)
			converted := convertAdmissionRequestToV1(orig)
			rt := convertAdmissionRequestToV1beta1(converted)
			if !reflect.DeepEqual(orig, rt) {
				t.Errorf("expected all request fields to be in converted object but found unaccounted for differences, diff:\n%s", cmp.Diff(orig, converted))
			}
		})
	}
}

func TestConvertAdmissionResponseToV1beta1(t *testing.T) {
	f := randfill.New()
	for i := 0; i < 100; i++ {
		t.Run(fmt.Sprintf("Run %d/100", i), func(t *testing.T) {
			orig := &v1.AdmissionResponse{}
			f.Fill(orig)
			converted := convertAdmissionResponseToV1beta1(orig)
			rt := convertAdmissionResponseToV1(converted)
			if !reflect.DeepEqual(orig, rt) {
				t.Errorf("expected all fields to be in converted object but found unaccounted for differences, diff:\n%s", cmp.Diff(orig, converted))
			}
		})
	}
}
