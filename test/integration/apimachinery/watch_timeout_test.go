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

package apimachinery

import (
	"context"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	kubectlproxy "k8s.io/kubectl/pkg/proxy"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestWatchClientTimeout(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	t.Run("direct", func(t *testing.T) {
		t.Logf("client at %s", s.URL)
		testWatchClientTimeouts(t, s.URL)
	})

	t.Run("reverse proxy", func(t *testing.T) {
		u, _ := url.Parse(s.URL)
		proxy := httputil.NewSingleHostReverseProxy(u)
		proxy.FlushInterval = -1
		proxyServer := httptest.NewServer(httputil.NewSingleHostReverseProxy(u))
		defer proxyServer.Close()

		t.Logf("client to %s, backend at %s", proxyServer.URL, s.URL)
		testWatchClientTimeouts(t, proxyServer.URL)
	})

	t.Run("kubectl proxy", func(t *testing.T) {
		kubectlProxyServer, err := kubectlproxy.NewServer("", "/", "/static/", nil, &restclient.Config{Host: s.URL, Timeout: 2 * time.Second}, 0)
		if err != nil {
			t.Fatal(err)
		}
		kubectlProxyListener, err := kubectlProxyServer.Listen("", 0)
		if err != nil {
			t.Fatal(err)
		}
		defer kubectlProxyListener.Close()
		go kubectlProxyServer.ServeOnListener(kubectlProxyListener)

		t.Logf("client to %s, backend at %s", kubectlProxyListener.Addr().String(), s.URL)
		testWatchClientTimeouts(t, "http://"+kubectlProxyListener.Addr().String())
	})
}

func testWatchClientTimeouts(t *testing.T, url string) {
	t.Run("timeout", func(t *testing.T) {
		testWatchClientTimeout(t, url, time.Second, 0)
	})
	t.Run("timeoutSeconds", func(t *testing.T) {
		testWatchClientTimeout(t, url, 0, time.Second)
	})
	t.Run("timeout+timeoutSeconds", func(t *testing.T) {
		testWatchClientTimeout(t, url, time.Second, time.Second)
	})
}

func testWatchClientTimeout(t *testing.T, serverURL string, timeout, timeoutSeconds time.Duration) {
	// client
	client, err := kubernetes.NewForConfig(&restclient.Config{Host: serverURL, Timeout: timeout})
	if err != nil {
		t.Fatal(err)
	}

	listCount := 0
	watchCount := 0
	stopCh := make(chan struct{})
	listWatch := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			t.Logf("listing (version=%s continue=%s)", options.ResourceVersion, options.Continue)
			listCount++
			if listCount > 1 {
				t.Errorf("listed more than once")
				close(stopCh)
			}
			return client.CoreV1().ConfigMaps(metav1.NamespaceAll).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			t.Logf("watching (version=%s)", options.ResourceVersion)
			if timeoutSeconds != 0 {
				timeout := int64(timeoutSeconds / time.Second)
				options.TimeoutSeconds = &timeout
			}
			watchCount++
			if watchCount > 1 {
				// success, restarted watch
				close(stopCh)
			}
			return client.CoreV1().ConfigMaps(metav1.NamespaceAll).Watch(context.TODO(), options)
		},
	}
	_, informer := cache.NewIndexerInformer(listWatch, &corev1.ConfigMap{}, 30*time.Minute, cache.ResourceEventHandlerFuncs{}, cache.Indexers{})
	informer.Run(stopCh)
	select {
	case <-stopCh:
	case <-time.After(time.Minute):
		t.Fatal("timeout")
	}
}
