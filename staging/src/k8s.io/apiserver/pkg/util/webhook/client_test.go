/*
Copyright 2021 The Kubernetes Authors.

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

type fakeAuthResolver struct {
	host string
}

func (sr fakeAuthResolver) ClientConfigFor(hostPort string) (*rest.Config, error) {
	config := &rest.Config{}
	config.Host = sr.host
	config.TLSClientConfig = rest.TLSClientConfig{
		Insecure:   true,
		ServerName: "127.0.0.1",
	}
	return config, nil
}

func (sr fakeAuthResolver) ClientConfigForService(serviceName, serviceNamespace string, servicePort int) (*rest.Config, error) {
	config := &rest.Config{}
	config.Host = sr.host
	config.TLSClientConfig = rest.TLSClientConfig{
		Insecure:   true,
		ServerName: "127.0.0.1",
	}
	return config, nil
}

func TestClientManagerNoProxyServices(t *testing.T) {
	// Create and start a fake proxy
	proxy, err := newTestServer(nil, nil, nil, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("Unexpected request on proxy: %v", r)
	})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	defer proxy.Close()
	t.Setenv("HTTPS_PROXY", proxy.URL)
	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("NO_PROXY", "")

	handler := func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello")
	}

	// Create and start a simple HTTPS server
	server, err := newTestServer(nil, nil, nil, handler)
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}
	defer server.Close()

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

	cconfig := ClientConfig{
		Name:     "test-client",
		CABundle: nil,
		Service: &ClientConfigService{
			Namespace: "test-ns",
			Name:      u.Hostname(),
			Port:      int32(port),
		},
	}

	client, err := cm.HookClient(cconfig)
	if err != nil {
		t.Fatal(err)
	}

	data, err := client.Get().AbsPath("/").DoRaw(context.TODO())
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if string(data) != "Hello" {
		t.Fatalf("unexpected response: %s", data)
	}

}
