/*
Copyright 2022 The Kubernetes Authors.

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
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
)

type fakeServiceResolver struct {
	url *url.URL
}

func (sr fakeServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return sr.url, nil
}

// The client builds a host url for services webhooks in the format: svcname.svcnamespace.svc
// The fake resolver allows to override the TLS configuration to use skip client verification
// and not fail due to the certificate errors, otherwise we should have to create custom certificates
// but certificate authentication is not really the goal of the test.
type fakeAuthResolver struct {
	host string
}

func (sr fakeAuthResolver) ClientConfigFor(hostPort string) (*rest.Config, error) {
	config := &rest.Config{}
	config.Host = sr.host
	config.TLSClientConfig = rest.TLSClientConfig{
		Insecure: true,
	}
	return config, nil
}

func (sr fakeAuthResolver) ClientConfigForService(serviceName, serviceNamespace string, servicePort int) (*rest.Config, error) {
	config := &rest.Config{}
	config.Host = sr.host
	config.TLSClientConfig = rest.TLSClientConfig{
		Insecure: true,
	}
	return config, nil
}

func TestClientManager(t *testing.T) {
	// Create and start a simple HTTPS server
	server, err := newTestServer(nil, nil, nil, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello")
	})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	defer server.Close()

	// Create a new ClientManager
	cm, err := NewClientManager(
		[]schema.GroupVersion{
			admissionv1.SchemeGroupVersion,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	p := u.Port()
	port, err := strconv.Atoi(p)
	if err != nil {
		t.Fatal(err)
	}

	cm.SetServiceResolver(&fakeServiceResolver{u})
	cm.SetAuthenticationInfoResolver(&fakeAuthResolver{server.URL})
	if err := cm.Validate(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		cconfig ClientConfig
	}{
		{
			name: "client using services",
			cconfig: ClientConfig{
				Name:     "test-client",
				CABundle: nil,
				Service: &ClientConfigService{
					Namespace: "test-ns",
					Name:      u.Hostname(),
					Port:      int32(port),
				},
			},
		},
		{
			name: "client using URL",
			cconfig: ClientConfig{
				Name:     "test-client2",
				URL:      server.URL,
				CABundle: nil,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client, err := cm.HookClient(tc.cconfig)
			if err != nil {
				t.Errorf("fail to get client: %v", err)
			}

			data, err := client.Get().AbsPath("/").DoRaw(context.TODO())
			if err != nil {
				t.Errorf("unexpected err: %v", err)
			}
			if string(data) != "Hello" {
				t.Errorf("unexpected response: %s", data)
			}
		})
	}

}
