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

package plugins

import (
	"net"
	"reflect"
	"testing"
)

func TestProxierAsEnvProvider(t *testing.T) {
	tests := []struct {
		p           *bearerProxier
		expectedEnv EnvList
	}{
		{
			p: &bearerProxier{
				listener: mockListener{},
				started:  false,
			},
			expectedEnv: EnvList{},
		},
		{
			p: &bearerProxier{
				listener: mockListener{},
				started:  true,
				bearer:   "123abc",
			},
			expectedEnv: EnvList{
				{"KUBECTL_PLUGINS_API_PROXY_ADDR", "127.0.0.1:8181"},
				{"KUBECTL_PLUGINS_API_PROXY_AUTH_TOKEN", "123abc"},
				{"KUBECTL_PLUGINS_API_PROXY_AUTH_HEADER", "Proxy-Authorization: Bearer 123abc"},
			},
		},
	}
	for _, test := range tests {
		if env, _ := test.p.Env(); !reflect.DeepEqual(env, test.expectedEnv) {
			t.Errorf("Incorrect env list provided by proxier, wanted %v, got %v %s", test.expectedEnv, env, []byte("127.0.0.1"))
		}
	}
}

type mockListener struct{}

func (m mockListener) Accept() (net.Conn, error) {
	return nil, nil
}

func (m mockListener) Close() error {
	return nil
}

func (m mockListener) Addr() net.Addr {
	return &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 8181,
	}
}
