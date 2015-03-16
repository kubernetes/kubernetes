/*
Copyright 2014 Google Inc. All rights reserved.

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

package envvars_test

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
)

func TestFromServices(t *testing.T) {
	sl := api.ServiceList{
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo-bar"},
				Spec: api.ServiceSpec{
					Port:     8080,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "1.2.3.4",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "abc-123"},
				Spec: api.ServiceSpec{
					Port:     8081,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "UDP",
					PortalIP: "5.6.7.8",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "q-u-u-x"},
				Spec: api.ServiceSpec{
					Port:     8082,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "9.8.7.6",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "svrc-portalip-none"},
				Spec: api.ServiceSpec{
					Port:     8082,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "None",
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "svrc-portalip-empty"},
				Spec: api.ServiceSpec{
					Port:     8082,
					Selector: map[string]string{"bar": "baz"},
					Protocol: "TCP",
					PortalIP: "",
				},
			},
		},
	}
	vars := envvars.FromServices(&sl)
	expected := []api.EnvVar{
		{Name: "FOO_BAR_SERVICE_HOST", Value: "1.2.3.4"},
		{Name: "FOO_BAR_SERVICE_PORT", Value: "8080"},
		{Name: "FOO_BAR_PORT", Value: "tcp://1.2.3.4:8080"},
		{Name: "FOO_BAR_PORT_8080_TCP", Value: "tcp://1.2.3.4:8080"},
		{Name: "FOO_BAR_PORT_8080_TCP_PROTO", Value: "tcp"},
		{Name: "FOO_BAR_PORT_8080_TCP_PORT", Value: "8080"},
		{Name: "FOO_BAR_PORT_8080_TCP_ADDR", Value: "1.2.3.4"},
		{Name: "ABC_123_SERVICE_HOST", Value: "5.6.7.8"},
		{Name: "ABC_123_SERVICE_PORT", Value: "8081"},
		{Name: "ABC_123_PORT", Value: "udp://5.6.7.8:8081"},
		{Name: "ABC_123_PORT_8081_UDP", Value: "udp://5.6.7.8:8081"},
		{Name: "ABC_123_PORT_8081_UDP_PROTO", Value: "udp"},
		{Name: "ABC_123_PORT_8081_UDP_PORT", Value: "8081"},
		{Name: "ABC_123_PORT_8081_UDP_ADDR", Value: "5.6.7.8"},
		{Name: "Q_U_U_X_SERVICE_HOST", Value: "9.8.7.6"},
		{Name: "Q_U_U_X_SERVICE_PORT", Value: "8082"},
		{Name: "Q_U_U_X_PORT", Value: "tcp://9.8.7.6:8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP", Value: "tcp://9.8.7.6:8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP_PROTO", Value: "tcp"},
		{Name: "Q_U_U_X_PORT_8082_TCP_PORT", Value: "8082"},
		{Name: "Q_U_U_X_PORT_8082_TCP_ADDR", Value: "9.8.7.6"},
	}
	if len(vars) != len(expected) {
		t.Errorf("Expected %d env vars, got: %+v", len(expected), vars)
		return
	}
	for i := range expected {
		if !reflect.DeepEqual(vars[i], expected[i]) {
			t.Errorf("expected %#v, got %#v", vars[i], expected[i])
		}
	}
}
