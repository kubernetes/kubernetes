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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/networking/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func TestIngressBackendConversion(t *testing.T) {
	scheme := runtime.NewScheme()
	assert.NoError(t, RegisterConversions(scheme))

	tests := map[string]struct {
		external v1beta1.IngressSpec
		internal networking.IngressSpec
	}{
		"service-port-number": {
			external: v1beta1.IngressSpec{
				Backend: &v1beta1.IngressBackend{
					ServiceName: "test-backend",
					ServicePort: intstr.FromInt32(8080),
				},
			},
			internal: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					Service: &networking.IngressServiceBackend{
						Name: "test-backend",
						Port: networking.ServiceBackendPort{
							Name:   "",
							Number: 8080,
						},
					},
				},
			},
		},
		"service-named-port": {
			external: v1beta1.IngressSpec{
				Backend: &v1beta1.IngressBackend{
					ServiceName: "test-backend",
					ServicePort: intstr.FromString("https"),
				},
			},
			internal: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					Service: &networking.IngressServiceBackend{
						Name: "test-backend",
						Port: networking.ServiceBackendPort{
							Name: "https",
						},
					},
				},
			},
		},
		"empty-service-name": {
			external: v1beta1.IngressSpec{
				Backend: &v1beta1.IngressBackend{
					ServiceName: "",
					ServicePort: intstr.FromString("https"),
				},
			},
			internal: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					Service: &networking.IngressServiceBackend{
						Name: "",
						Port: networking.ServiceBackendPort{
							Name: "https",
						},
					},
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			convertedInternal := networking.IngressSpec{}
			require.NoError(t,
				Convert_v1beta1_IngressSpec_To_networking_IngressSpec(&test.external, &convertedInternal, nil))
			assert.Equal(t, test.internal, convertedInternal, "v1beta1.IngressSpec -> networking.IngressSpec")

			convertedV1beta1 := v1beta1.IngressSpec{}
			require.NoError(t,
				Convert_networking_IngressSpec_To_v1beta1_IngressSpec(&test.internal, &convertedV1beta1, nil))
			assert.Equal(t, test.external, convertedV1beta1, "networking.IngressSpec -> v1beta1.IngressSpec")
		})
	}
}
