/*
Copyright 2025 The Kubernetes Authors.

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

package oidc

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	utilsoidc "k8s.io/kubernetes/test/utils/oidc"
)

func runEgressProxy(t testing.TB, udsName string, ready chan<- struct{}) {
	t.Helper()

	l, err := net.Listen("unix", udsName)
	if err != nil {
		t.Errorf("unexpected UDS error: %v", err)
		return
	}

	var called atomic.Bool
	server := http.Server{Handler: utilsoidc.NewHTTPConnectProxyHandler(t, &called)}

	t.Cleanup(func() {
		if !called.Load() {
			t.Errorf("egress proxy was not called")
		}

		err := server.Shutdown(context.Background())
		if err != nil && !utilnet.IsProbableEOF(err) {
			t.Logf("shutdown exit error: %v", err)
		}
	})

	var once sync.Once
	readyCheckClient := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", udsName)
			},
		},
	}
	go func() {
		if err := wait.PollUntilContextCancel(t.Context(), time.Second, false, func(ctx context.Context) (bool, error) {
			resp, err := readyCheckClient.Get("http://host.does.not.matter/ready")
			if err != nil {
				t.Logf("egress proxy error: %v", err)
				return false, nil
			}
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Logf("egress proxy unexpected status code: %v", resp.StatusCode)
				return false, nil
			}
			once.Do(func() { close(ready) })
			return true, nil
		}); err != nil {
			t.Errorf("egress proxy is not ready: %v", err)
		}
	}()

	err = server.Serve(l)
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		t.Logf("egress exit error: %v", err)
	}
}
