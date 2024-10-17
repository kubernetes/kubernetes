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

package util

import (
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

// GlobalUnicastHostIP returns a global unicast IP address that can be used for
// httptest servers. In particular, this guarantees that the IP address is not
// 127.0.0.1 or ::1.
func GlobalUnicastHostIP(t *testing.T) net.IP {
	addrs, err := net.InterfaceAddrs()
	require.NoError(t, err)
	for _, addr := range addrs {
		ip, _, err := net.ParseCIDR(addr.String())
		require.NoError(t, err)
		if !ip.IsGlobalUnicast() {
			continue
		}

		// smoke test to make sure we can run a test server on it
		s := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		}))
		s.Listener, err = net.Listen("tcp", net.JoinHostPort(ip.String(), "0"))
		if err != nil {
			t.Logf("skipping %s as global unicast address for httptest server: %v", ip, err)
			continue
		}
		s.Start()
		defer s.Close()

		// smoke test that we can connect to it
		resp, err := http.Get(s.URL)
		if err != nil {
			t.Logf("skipping %s as global unicast address for httptest server because we cannot connect: %v", ip, err)
			continue
		}
		resp.Body.Close()

		t.Logf("using %s as global unicast address for httptest server", ip)
		return ip
	}
	t.Fatal("no global unicast host IP found that can be used for httptest server")
	return nil
}
