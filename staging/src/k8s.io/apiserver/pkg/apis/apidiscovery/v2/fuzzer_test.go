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

package v2_test

import (
	"reflect"
	"testing"

	v2 "k8s.io/api/apidiscovery/v2"
	v2beta1 "k8s.io/api/apidiscovery/v2beta1"
	v2scheme "k8s.io/apiserver/pkg/apis/apidiscovery/v2"
	v2beta1scheme "k8s.io/apiserver/pkg/apis/apidiscovery/v2beta1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/require"
)

func TestConversionRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	err := v2beta1scheme.AddToScheme(scheme)
	require.NoError(t, err)
	err = v2scheme.AddToScheme(scheme)
	require.NoError(t, err)
	err = v2scheme.RegisterConversions(scheme)
	require.NoError(t, err)

	fuzzer := fuzz.NewWithSeed(2374375)

	// v2 -> v2beta1 -> v2
	for i := 0; i < 100; i++ {
		expected := &v2.APIGroupDiscoveryList{}
		fuzzer.Fuzz(expected)
		expected.TypeMeta = metav1.TypeMeta{
			Kind:       "APIGroupDiscoveryList",
			APIVersion: "apidiscovery.k8s.io/v2",
		}
		o, err := scheme.ConvertToVersion(expected, v2beta1.SchemeGroupVersion)
		require.NoError(t, err)
		v2beta1Type := o.(*v2beta1.APIGroupDiscoveryList)

		o2, err := scheme.ConvertToVersion(v2beta1Type, v2.SchemeGroupVersion)
		require.NoError(t, err)
		actual := o2.(*v2.APIGroupDiscoveryList)

		if !reflect.DeepEqual(expected, actual) {
			t.Error(cmp.Diff(expected, actual))
		}
	}

	// v2beta1 -> v2 -> v2beta1
	for i := 0; i < 100; i++ {
		expected := &v2beta1.APIGroupDiscoveryList{}
		fuzzer.Fuzz(expected)
		expected.TypeMeta = metav1.TypeMeta{
			Kind:       "APIGroupDiscoveryList",
			APIVersion: "apidiscovery.k8s.io/v2beta1",
		}
		o, err := scheme.ConvertToVersion(expected, v2.SchemeGroupVersion)
		require.NoError(t, err)
		v2Type := o.(*v2.APIGroupDiscoveryList)

		o2, err := scheme.ConvertToVersion(v2Type, v2beta1.SchemeGroupVersion)
		require.NoError(t, err)
		actual := o2.(*v2beta1.APIGroupDiscoveryList)

		if !reflect.DeepEqual(expected, actual) {
			t.Error(cmp.Diff(expected, actual))
		}
	}
}
