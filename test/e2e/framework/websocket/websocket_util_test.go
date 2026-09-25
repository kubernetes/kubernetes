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

package websocket

import (
	"net/http/httptest"
	"net/url"
	"testing"

	"golang.org/x/net/websocket"

	restclient "k8s.io/client-go/rest"
)

// TestOpenWebSocketForURLReusesURL checks that the same *url.URL can be passed
// to OpenWebSocketForURL more than once, as tests do when they retry a
// websocket read, and that the helper does not modify the caller's URL.
func TestOpenWebSocketForURLReusesURL(t *testing.T) {
	server := httptest.NewTLSServer(websocket.Handler(func(ws *websocket.Conn) {
		_, _ = ws.Write([]byte("hello"))
	}))
	defer server.Close()

	config := &restclient.Config{TLSClientConfig: restclient.TLSClientConfig{Insecure: true}}
	u, err := url.Parse(server.URL + "/log")
	if err != nil {
		t.Fatal(err)
	}

	for attempt := 1; attempt <= 3; attempt++ {
		ws, err := OpenWebSocketForURL(u, config, []string{"binary.k8s.io"})
		if err != nil {
			t.Fatalf("attempt %d: unexpected error: %v", attempt, err)
		}
		var msg []byte
		if err := websocket.Message.Receive(ws, &msg); err != nil {
			t.Fatalf("attempt %d: unexpected error receiving: %v", attempt, err)
		}
		_ = ws.Close()
		if string(msg) != "hello" {
			t.Errorf("attempt %d: expected %q, got %q", attempt, "hello", string(msg))
		}
		if u.Scheme != "https" {
			t.Errorf("attempt %d: the caller's URL scheme was changed to %q", attempt, u.Scheme)
		}
	}
}
