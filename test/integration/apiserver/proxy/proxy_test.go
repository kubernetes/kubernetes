/*
Copyright The Kubernetes Authors.

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

package proxy

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
)

func buildWebsocketConfig(u *url.URL, config *restclient.Config) (*websocket.Config, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, err
	}
	if u.Scheme == "https" {
		u.Scheme = "wss"
	} else {
		u.Scheme = "ws"
	}
	cfg, err := websocket.NewConfig(u.String(), "http://localhost")
	if err != nil {
		return nil, err
	}
	cfg.Header = make(http.Header)
	if len(config.BearerToken) > 0 {
		cfg.Header.Set("Authorization", "Bearer "+config.BearerToken)
	}
	cfg.TlsConfig = tlsConfig
	return cfg, nil
}

func TestProxyForwarding(t *testing.T) {
	ns := "default"
	fakeHTTPPodIP := "10.0.0.10"
	fakeHTTPSPodIP := "10.0.0.20"

	// 1. Setup HTTP Backend
	receivedHTTPQuery := make(chan string, 10)
	httpBackendHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Upgrade") == "websocket" {
			receivedHTTPQuery <- r.URL.RawQuery
			websocket.Handler(func(ws *websocket.Conn) {
				_, _ = ws.Write([]byte("hello-ws"))
			}).ServeHTTP(w, r)
			return
		}
		receivedHTTPQuery <- r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("hello-http"))
	})

	httpBackendServer := httptest.NewServer(httpBackendHandler)
	defer httpBackendServer.Close()

	httpBackendURL, err := url.Parse(httpBackendServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	httpBackendPort := httpBackendURL.Port()
	httpBackendPortVal, err := strconv.Atoi(httpBackendPort)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Setup HTTPS Backend
	receivedHTTPSQuery := make(chan string, 10)
	httpsBackendHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Upgrade") == "websocket" {
			receivedHTTPSQuery <- r.URL.RawQuery
			websocket.Handler(func(ws *websocket.Conn) {
				_, _ = ws.Write([]byte("hello-wss"))
			}).ServeHTTP(w, r)
			return
		}
		receivedHTTPSQuery <- r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("hello-https"))
	})

	httpsBackendServer := httptest.NewTLSServer(httpsBackendHandler)
	defer httpsBackendServer.Close()

	httpsBackendURL, err := url.Parse(httpsBackendServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	httpsBackendPort := httpsBackendURL.Port()
	httpsBackendPortVal, err := strconv.Atoi(httpsBackendPort)
	if err != nil {
		t.Fatal(err)
	}

	originalDialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	proxyDialer := func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return originalDialer.DialContext(ctx, network, addr)
		}
		switch host {
		case fakeHTTPPodIP:
			if port != httpBackendPort {
				t.Errorf("Expected dialer to dial port %s for fakeHTTPPodIP, got %s", httpBackendPort, port)
			}
			addr = net.JoinHostPort("127.0.0.1", httpBackendPort)
		case fakeHTTPSPodIP:
			if port != httpsBackendPort {
				t.Errorf("Expected dialer to dial port %s for fakeHTTPSPodIP, got %s", httpsBackendPort, port)
			}
			addr = net.JoinHostPort("127.0.0.1", httpsBackendPort)
		}
		return originalDialer.DialContext(ctx, network, addr)
	}

	setup := framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
		},
		ModifyServerConfig: func(config *controlplane.Config) {
			config.ControlPlane.Extra.ProxyTransport.DialContext = proxyDialer
		},
	}
	_, clientConfig, tearDown := framework.StartTestServer(context.TODO(), t, setup)
	defer tearDown()

	clientset, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// Create Pod pointing to HTTP Backend with a fake global unicast IP
	httpPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "http-pod", Namespace: ns},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "nginx", Image: "nginx"}},
		},
	}
	createdHTTPPod, err := clientset.CoreV1().Pods(ns).Create(context.TODO(), httpPod, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	createdHTTPPod.Status.PodIP = fakeHTTPPodIP
	createdHTTPPod.Status.Phase = corev1.PodRunning
	_, err = clientset.CoreV1().Pods(ns).UpdateStatus(context.TODO(), createdHTTPPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Create Pod pointing to HTTPS Backend with a fake global unicast IP
	httpsPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "https-pod", Namespace: ns},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "nginx", Image: "nginx"}},
		},
	}
	createdHTTPSPod, err := clientset.CoreV1().Pods(ns).Create(context.TODO(), httpsPod, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	createdHTTPSPod.Status.PodIP = fakeHTTPSPodIP
	createdHTTPSPod.Status.Phase = corev1.PodRunning
	_, err = clientset.CoreV1().Pods(ns).UpdateStatus(context.TODO(), createdHTTPSPod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Create Services pointing to our pods
	httpService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "http-service", Namespace: ns},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt(httpBackendPortVal),
				},
			},
		},
	}
	_, err = clientset.CoreV1().Services(ns).Create(context.TODO(), httpService, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	//nolint:staticcheck // SA1019 Endpoints is deprecated, but still used in the apiserver
	httpEndpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "http-service", Namespace: ns},
		//nolint:staticcheck // SA1019 EndpointSubset is deprecated, but still used in the apiserver
		Subsets: []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{
					{
						IP: fakeHTTPPodIP,
						TargetRef: &corev1.ObjectReference{
							Kind:      "Pod",
							Name:      "http-pod",
							Namespace: ns,
						},
					},
				},
				Ports: []corev1.EndpointPort{
					{
						Name: "http",
						Port: int32(httpBackendPortVal),
					},
				},
			},
		},
	}
	_, err = clientset.CoreV1().Endpoints(ns).Create(context.TODO(), httpEndpoints, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	httpsService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "https-service", Namespace: ns},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Name:       "https",
					Port:       443,
					TargetPort: intstr.FromInt(httpsBackendPortVal),
				},
			},
		},
	}
	_, err = clientset.CoreV1().Services(ns).Create(context.TODO(), httpsService, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	//nolint:staticcheck // SA1019 Endpoints is deprecated, but still used in the apiserver
	httpsEndpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "https-service", Namespace: ns},
		//nolint:staticcheck // SA1019 EndpointSubset is deprecated, but still used in the apiserver
		Subsets: []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{
					{
						IP: fakeHTTPSPodIP,
						TargetRef: &corev1.ObjectReference{
							Kind:      "Pod",
							Name:      "https-pod",
							Namespace: ns,
						},
					},
				},
				Ports: []corev1.EndpointPort{
					{
						Name: "https",
						Port: int32(httpsBackendPortVal),
					},
				},
			},
		},
	}
	_, err = clientset.CoreV1().Endpoints(ns).Create(context.TODO(), httpsEndpoints, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Helper for constructing clients
	httpClient, err := restclient.HTTPClientFor(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// Subtest 1: HTTP Pod Proxy
	t.Run("HTTP", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/pods/http-pod:" + httpBackendPort + "/proxy/?foo=bar"
		resp, err := httpClient.Get(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", resp.StatusCode)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != "hello-http" {
			t.Errorf("expected 'hello-http', got %q", string(body))
		}

		select {
		case q := <-receivedHTTPQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 2: HTTPS Pod Proxy
	t.Run("HTTPS", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/pods/https:https-pod:" + httpsBackendPort + "/proxy/?foo=bar"
		resp, err := httpClient.Get(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", resp.StatusCode)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != "hello-https" {
			t.Errorf("expected 'hello-https', got %q", string(body))
		}

		select {
		case q := <-receivedHTTPSQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 3: WS (WebSocket Unsecured) Pod Proxy
	t.Run("WS", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/pods/http-pod:" + httpBackendPort + "/proxy/?foo=bar"
		u, err := url.Parse(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}

		wsc, err := buildWebsocketConfig(u, clientConfig)
		if err != nil {
			t.Fatal(err)
		}

		wsConn, err := websocket.DialConfig(wsc)
		if err != nil {
			t.Fatalf("websocket Dial failed: %v", err)
		}
		defer wsConn.Close() //nolint:errcheck

		var msg = make([]byte, 512)
		n, err := wsConn.Read(msg)
		if err != nil {
			t.Fatal(err)
		}
		if string(msg[:n]) != "hello-ws" {
			t.Errorf("expected 'hello-ws', got %q", string(msg[:n]))
		}

		select {
		case q := <-receivedHTTPQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 4: WSS (WebSocket Secured) Pod Proxy
	t.Run("WSS", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/pods/https:https-pod:" + httpsBackendPort + "/proxy/?foo=bar"
		u, err := url.Parse(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}

		wsc, err := buildWebsocketConfig(u, clientConfig)
		if err != nil {
			t.Fatal(err)
		}

		wsConn, err := websocket.DialConfig(wsc)
		if err != nil {
			t.Fatalf("websocket Dial failed: %v", err)
		}
		defer wsConn.Close() //nolint:errcheck

		var msg = make([]byte, 512)
		n, err := wsConn.Read(msg)
		if err != nil {
			t.Fatal(err)
		}
		if string(msg[:n]) != "hello-wss" {
			t.Errorf("expected 'hello-wss', got %q", string(msg[:n]))
		}

		select {
		case q := <-receivedHTTPSQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 5: HTTP Service Proxy
	t.Run("HTTPService", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/services/http-service:80/proxy/?foo=bar"
		resp, err := httpClient.Get(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", resp.StatusCode)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != "hello-http" {
			t.Errorf("expected 'hello-http', got %q", string(body))
		}

		select {
		case q := <-receivedHTTPQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 6: HTTPS Service Proxy
	t.Run("HTTPSService", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/services/https:https-service:443/proxy/?foo=bar"
		resp, err := httpClient.Get(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close() //nolint:errcheck

		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", resp.StatusCode)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != "hello-https" {
			t.Errorf("expected 'hello-https', got %q", string(body))
		}

		select {
		case q := <-receivedHTTPSQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 7: WS Service Proxy
	t.Run("WSService", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/services/http-service:80/proxy/?foo=bar"
		u, err := url.Parse(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}

		wsc, err := buildWebsocketConfig(u, clientConfig)
		if err != nil {
			t.Fatal(err)
		}

		wsConn, err := websocket.DialConfig(wsc)
		if err != nil {
			t.Fatalf("websocket Dial failed: %v", err)
		}
		defer wsConn.Close() //nolint:errcheck

		var msg = make([]byte, 512)
		n, err := wsConn.Read(msg)
		if err != nil {
			t.Fatal(err)
		}
		if string(msg[:n]) != "hello-ws" {
			t.Errorf("expected 'hello-ws', got %q", string(msg[:n]))
		}

		select {
		case q := <-receivedHTTPQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})

	// Subtest 8: WSS Service Proxy
	t.Run("WSSService", func(t *testing.T) {
		proxyPath := "/api/v1/namespaces/" + ns + "/services/https:https-service:443/proxy/?foo=bar"
		u, err := url.Parse(clientConfig.Host + proxyPath)
		if err != nil {
			t.Fatal(err)
		}

		wsc, err := buildWebsocketConfig(u, clientConfig)
		if err != nil {
			t.Fatal(err)
		}

		wsConn, err := websocket.DialConfig(wsc)
		if err != nil {
			t.Fatalf("websocket Dial failed: %v", err)
		}
		defer wsConn.Close() //nolint:errcheck

		var msg = make([]byte, 512)
		n, err := wsConn.Read(msg)
		if err != nil {
			t.Fatal(err)
		}
		if string(msg[:n]) != "hello-wss" {
			t.Errorf("expected 'hello-wss', got %q", string(msg[:n]))
		}

		select {
		case q := <-receivedHTTPSQuery:
			if !strings.Contains(q, "foo=bar") {
				t.Errorf("expected query 'foo=bar', got %q", q)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for query string")
		}
	})
}
