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

package banflunder_test

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/sample-apiserver/pkg/admission/plugin/banflunder"
	"k8s.io/sample-apiserver/pkg/admission/wardleinitializer"
	wardle "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	"k8s.io/sample-apiserver/pkg/generated/clientset/versioned/fake"
	informers "k8s.io/sample-apiserver/pkg/generated/informers/externalversions"
)

// TestBanfluderAdmissionPlugin tests various test cases against
// ban flunder admission plugin
func TestBanflunderAdmissionPlugin(t *testing.T) {
	var scenarios = []struct {
		informersOutput        wardle.FischerList
		admissionInput         wardle.Flunder
		admissionInputKind     schema.GroupVersionKind
		admissionInputResource schema.GroupVersionResource
		admissionMustFail      bool
	}{
		// scenario 1:
		// a flunder with a name that appears on a list of disallowed flunders must be banned
		{
			informersOutput: wardle.FischerList{
				Items: []wardle.Fischer{
					{DisallowedFlunders: []string{"badname"}},
				},
			},
			admissionInput: wardle.Flunder{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "badname",
					Namespace: "",
				},
			},
			admissionInputKind:     wardle.SchemeGroupVersion.WithKind("Flunder").GroupKind().WithVersion("version"),
			admissionInputResource: wardle.Resource("flunders").WithVersion("version"),
			admissionMustFail:      true,
		},
		// scenario 2:
		// a flunder with a name that does not appear on a list of disallowed flunders must be admitted
		{
			informersOutput: wardle.FischerList{
				Items: []wardle.Fischer{
					{DisallowedFlunders: []string{"badname"}},
				},
			},
			admissionInput: wardle.Flunder{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "goodname",
					Namespace: "",
				},
			},
			admissionInputKind:     wardle.SchemeGroupVersion.WithKind("Flunder").GroupKind().WithVersion("version"),
			admissionInputResource: wardle.Resource("flunders").WithVersion("version"),
			admissionMustFail:      false,
		},
		// scenario 3:
		// a flunder with a name that appears on a list of disallowed flunders would be banned
		// but the kind passed in is not a flunder thus the whole request is accepted
		{
			informersOutput: wardle.FischerList{
				Items: []wardle.Fischer{
					{DisallowedFlunders: []string{"badname"}},
				},
			},
			admissionInput: wardle.Flunder{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "badname",
					Namespace: "",
				},
			},
			admissionInputKind:     wardle.SchemeGroupVersion.WithKind("NotFlunder").GroupKind().WithVersion("version"),
			admissionInputResource: wardle.Resource("notflunders").WithVersion("version"),
			admissionMustFail:      false,
		},
	}

	for index, scenario := range scenarios {
		func() {
			// prepare
			cs := &fake.Clientset{}
			cs.AddReactor("list", "fischers", func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, &scenario.informersOutput, nil
			})
			informersFactory := informers.NewSharedInformerFactory(cs, 5*time.Minute)

			target, err := banflunder.New()
			if err != nil {
				t.Fatalf("scenario %d: failed to create banflunder admission plugin due to = %v", index, err)
			}

			targetInitializer := wardleinitializer.New(informersFactory)
			targetInitializer.Initialize(target)

			err = admission.ValidateInitialization(target)
			if err != nil {
				t.Fatalf("scenario %d: failed to initialize banflunder admission plugin due to =%v", index, err)
			}

			stop := make(chan struct{})
			defer close(stop)
			informersFactory.Start(stop)
			informersFactory.WaitForCacheSync(stop)

			// act
			err = target.Admit(admission.NewAttributesRecord(
				&scenario.admissionInput,
				nil,
				scenario.admissionInputKind,
				scenario.admissionInput.ObjectMeta.Namespace,
				"",
				scenario.admissionInputResource,
				"",
				admission.Create,
				&metav1.CreateOptions{},
				false,
				nil),
				nil,
			)

			// validate
			if scenario.admissionMustFail && err == nil {
				t.Errorf("scenario %d: expected an error but got nothing", index)
			}

			if !scenario.admissionMustFail && err != nil {
				t.Errorf("scenario %d: banflunder admission plugin returned unexpected error = %v", index, err)
			}
		}()
	}
}
