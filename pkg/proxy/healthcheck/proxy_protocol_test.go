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

package healthcheck

import (
	"bufio"
	"encoding/binary"
	"net"
	"net/http"
	"slices"
	"testing"
	"time"
)

// proxyV2Header builds a PROXY v2 header. cmd is 0x0 (LOCAL: connection made
// by the proxy itself, e.g. a health check; addresses are ignored) or 0x1
// (PROXY: relayed on behalf of a client; addr carries its source/destination).
func proxyV2Header(cmd byte, addr []byte) []byte {
	h := slices.Clone(proxyV2Signature)
	h = append(h, 0x20|cmd, 0x11, 0, 0) // version 2 | cmd, AF_INET | STREAM, length placeholder
	binary.BigEndian.PutUint16(h[14:16], uint16(len(addr)))
	return append(h, addr...)
}

func TestProxyProtocolListener(t *testing.T) {
	ipv4Addrs := []byte{10, 0, 0, 1, 10, 0, 0, 2, 0x1f, 0x90, 0x7a, 0x69}
	tooLarge := proxyV2Header(0x1, nil)
	binary.BigEndian.PutUint16(tooLarge[14:16], proxyV2MaxAddrLen+1)
	badVersion := proxyV2Header(0x1, ipv4Addrs)
	badVersion[12] = 0x11
	badCommand := proxyV2Header(0x2, ipv4Addrs)

	testCases := []struct {
		name     string
		preamble []byte
		wantOK   bool
	}{
		{name: "plain HTTP", wantOK: true},
		{name: "PROXY v2 PROXY command", preamble: proxyV2Header(0x1, ipv4Addrs), wantOK: true},
		{name: "PROXY v2 LOCAL command, no addresses", preamble: proxyV2Header(0x0, nil), wantOK: true},
		{name: "PROXY v2 address block too large", preamble: tooLarge},
		{name: "PROXY v2 wrong version", preamble: badVersion},
		{name: "PROXY v2 unknown command", preamble: badCommand},
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})}
	go func() { _ = srv.Serve(proxyProtocolListener{Listener: ln}) }()
	t.Cleanup(func() { _ = srv.Close() })

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			conn, err := net.Dial("tcp", ln.Addr().String())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = conn.Close() }()
			_ = conn.SetDeadline(time.Now().Add(5 * time.Second))

			req := tc.preamble
			req = append(req, []byte("GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")...)
			if _, err := conn.Write(req); err != nil {
				t.Fatal(err)
			}
			resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
			if !tc.wantOK {
				// net/http turns the read error into a 400 or drops the connection.
				if err == nil && resp.StatusCode == http.StatusOK {
					t.Fatal("expected malformed PROXY header to be rejected, got 200")
				}
				return
			}
			if err != nil {
				t.Fatalf("reading response: %v", err)
			}
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("expected 200, got %d", resp.StatusCode)
			}
		})
	}
}
