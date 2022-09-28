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
	"reflect"
	"testing"
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
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
	defaultClass1 := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "default1",
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}
	defaultClass2 := &networkingv1.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default2",
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}
	// Class that has explicit default = false
	classWithFalseDefault := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault1",
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "false",
			},
		},
	}
	// Class with missing default annotation (=non-default)
	classWithNoDefault := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
		},
	}
	// Class with empty default annotation (=non-default)
	classWithEmptyDefault := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "",
			},
		},
	}

	defaultClassWithCreateTime1 := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:              "default1",
			CreationTimestamp: metav1.NewTime(time.Date(2022, time.Month(1), 1, 0, 0, 0, 1, time.UTC)),
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}
	defaultClassWithCreateTime2 := &networkingv1.IngressClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "IngressClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:              "default2",
			CreationTimestamp: metav1.NewTime(time.Date(2022, time.Month(1), 1, 0, 0, 0, 0, time.UTC)),
			Annotations: map[string]string{
				networkingv1.AnnotationIsDefaultIngressClass: "true",
			},
		},
	}

	testCases := []struct {
		name            string
		classes         []*networkingv1.IngressClass
		classField      *string
		classAnnotation *string
		expectedClass   *string
		expectedError   error
	}{
		{
			name:            "no default, no modification of Ingress",
			classes:         []*networkingv1.IngressClass{classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   nil,
			expectedError:   nil,
		},
		{
			name:            "one default, modify Ingress with class=nil",
			classes:         []*networkingv1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(defaultClass1.Name),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class field=''",
			classes:         []*networkingv1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr(""),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(""),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class field='foo'",
			classes:         []*networkingv1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr("foo"),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr("foo"),
			expectedError:   nil,
		},
		{
			name:            "one default, no modification of Ingress with class annotation='foo'",
			classes:         []*networkingv1.IngressClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: utilpointer.StringPtr("foo"),
			expectedClass:   nil,
			expectedError:   nil,
		},
		{
			name:            "two defaults with the same creation time, choose the one with the lower name",
			classes:         []*networkingv1.IngressClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(defaultClass1.Name),
			expectedError:   nil,
		},
		{
			name:            "two defaults, no modification with Ingress with class field=''",
			classes:         []*networkingv1.IngressClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      utilpointer.StringPtr(""),
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(""),
			expectedError:   nil,
		},
		{
			name:            "two defaults, choose the one with the newer creation time",
			classes:         []*networkingv1.IngressClass{defaultClassWithCreateTime1, defaultClassWithCreateTime2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			classField:      nil,
			classAnnotation: nil,
			expectedClass:   utilpointer.StringPtr(defaultClassWithCreateTime1.Name),
			expectedError:   nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctrl := newPlugin()
			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			ctrl.SetExternalKubeInformerFactory(informerFactory)
			for _, c := range testCase.classes {
				informerFactory.Networking().V1().IngressClasses().Informer().GetStore().Add(c)
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
				networkingv1.Resource("ingresses").WithVersion("version"),
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
