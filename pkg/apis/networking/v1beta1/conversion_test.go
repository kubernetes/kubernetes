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

package v1beta1

import (
	"github.com/stretchr/testify/assert"
	"k8s.io/api/networking/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/networking"
	"testing"
)

func TestIngressBackendConversion(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, RegisterConversions(scheme))

	v1beta1Backend := v1beta1.IngressBackend{
		ServiceName: "test-backend",
		ServicePort: intstr.FromInt(8080),
	}

	v1beta1Spec := v1beta1.IngressSpec{Backend: &v1beta1Backend}

	internalSpec := networking.IngressSpec{}

	assert.NoError(t, scheme.Convert(&v1beta1Spec, &internalSpec, nil))

	if internalSpec.DefaultBackend.ServiceName != "test-backend" {
		t.Errorf("Convert v1beta1.Backend to DefaultBackend failed. Expected =%v but found %v", v1beta1Spec.Backend.ServiceName, internalSpec.DefaultBackend.ServiceName)
	}

	internalDefaultBackend := networking.IngressBackend{
		ServiceName: "test-backend",
		ServicePort: intstr.FromInt(8080),
	}

	internalSpec = networking.IngressSpec{DefaultBackend: &internalDefaultBackend}
	v1beta1Spec = v1beta1.IngressSpec{}

	assert.NoError(t, scheme.Convert(&internalSpec, &v1beta1Spec, nil))

	if v1beta1Spec.Backend.ServiceName != "test-backend" {
		t.Errorf("Convert DefaultBackend to v1beta1.Backend failed. Expected =%v but found %v", internalSpec.DefaultBackend.ServiceName, v1beta1Spec.Backend.ServiceName)
	}
}
