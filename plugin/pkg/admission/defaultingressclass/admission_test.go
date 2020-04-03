/*
Copyright 2020 The Kubernetes Authors.

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

package defaultingressclass

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	"k8s.io/client-go/informers"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/controller"
	utilpointer "k8s.io/utils/pointer"
)

func TestAdmission(t *testing.T) {
	defaultClass1 := &networkingv1beta1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "default1",
			Annotations: map[string]string{
				networkingv1beta1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}
	defaultClass2 := &networkingv1beta1.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default2",
			Annotations: map[string]string{
				networkingv1beta1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}
	// Class that has explicit default = false
	classWithFalseDefault := &networkingv1beta1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault1",
			Annotations: map[string]string{
				networkingv1beta1.AnnotationIsDefaultIngressClass: "false",
			},
		},
	}
	// Class with missing default annotation (=non-default)
	classWithNoDefault := &networkingv1beta1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
		},
	}
	// Class with empty default annotation (=non-default)
	classWithEmptyDefault := &networkingv1beta1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
			Annotations: map[string]string{
				networkingv1beta1.AnnotationIsDefaultIngressClass: "",
			},
		},
	}

	testCases := []struct {
		name            string
		classes         []*networkingv1beta1.IngressClass
		classField      *string
		classAnnotation *string
		expectedClass   *string
		expectedError   error
	}{
		{
			name:            "no default, no modification of Ingress",
			classes:         []*networkingv1beta1.IngressClass{classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   nil,
			expectedError:   nil,
		},
		{
			name:            "one default, modify Ingress with class=nil",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(defaultClass1.Name),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class field=''",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr(""),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(""),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class field='foo'",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr("foo"),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr("foo"),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class annotation='foo'",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: utilpointer.StringPtr("foo"),
			expectedClass:   nil,
			expectedError:   nil,
		},
		{
			name:            "two defaults, error with Ingress with class field=nil",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   nil,
			expectedError:   errors.NewForbidden(networkingv1beta1.Resource("ingresses"), "testing", errors.NewInternalError(fmt.Errorf("2 default IngressClasses were found, only 1 allowed"))),
		},
		{
			name:            "two defaults, no modification with Ingress with class field=''",
			classes:         []*networkingv1beta1.IngressClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr(""),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(""),
			expectedError:   nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctrl := newPlugin()
			ctrl.defaultIngressClassEnabled = true
			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			ctrl.SetExternalKubeInformerFactory(informerFactory)
			for _, c := range testCase.classes {
				informerFactory.Networking().V1beta1().IngressClasses().Informer().GetStore().Add(c)
			}

			ingress := &networking.Ingress{ObjectMeta: metav1.ObjectMeta{Name: "testing", Namespace: "testing"}}
			if testCase.classField != nil {
				ingress.Spec.IngressClassName = testCase.classField
			}
			if testCase.classAnnotation != nil {
				ingress.Annotations = map[string]string{networkingv1beta1.AnnotationIngressClass: *testCase.classAnnotation}
			}

			attrs := admission.NewAttributesRecord(
				ingress, // new object
				nil,     // old object
				api.Kind("Ingress").WithVersion("version"),
				ingress.Namespace,
				ingress.Name,
				networkingv1beta1.Resource("ingresses").WithVersion("version"),
				"", // subresource
				admission.Create,
				&metav1.CreateOptions{},
				false, // dryRun
				nil,   // userInfo
			)

			err := admissiontesting.WithReinvocationTesting(t, ctrl).Admit(context.TODO(), attrs, nil)
			if !reflect.DeepEqual(err, testCase.expectedError) {
				t.Errorf("Expected error: %v, got %v", testCase.expectedError, err)
			}
			if !reflect.DeepEqual(testCase.expectedClass, ingress.Spec.IngressClassName) {
				t.Errorf("Expected class name %+v, got %+v", *testCase.expectedClass, ingress.Spec.IngressClassName)
			}
		})
	}
}
