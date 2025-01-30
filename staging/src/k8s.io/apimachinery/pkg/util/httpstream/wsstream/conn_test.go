/*
Copyright 2015 The Kubernetes Authors.

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

package wsstream

import (
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/websocket"
)

func newServer(handler http.Handler) (*httptest.Server, string) {
	server := httptest.NewServer(handler)
	serverAddr := server.Listener.Addr().String()
	return server, serverAddr
}

func TestRawConn(t *testing.T) {
	channels := []ChannelType{ReadWriteChannel, ReadWriteChannel, IgnoreChannel, ReadChannel, WriteChannel}
	conn := NewConn(NewDefaultChannelProtocols(channels))

	s, addr := newServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conn.Open(w, req)
	}))
	defer s.Close()

	client, err := websocket.Dial("ws://"+addr, "", "http://localhost/")
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	<-conn.ready
	wg := sync.WaitGroup{}

	// verify we can read a client write
	wg.Add(1)
	go func() {
		defer wg.Done()
		data, err := io.ReadAll(conn.channels[0])
		if err != nil {
			t.Error(err)
			return
		}
		if !reflect.DeepEqual(data, []byte("client")) {
			t.Errorf("unexpected server read: %v", data)
		}
	}()

	if n, err := client.Write(append([]byte{0}, []byte("client")...)); err != nil || n != 7 {
		t.Fatalf("%d: %v", n, err)
	}

	// verify we can read a server write
	wg.Add(1)
	go func() {
		defer wg.Done()
		if n, err := conn.channels[1].Write([]byte("server")); err != nil && n != 6 {
			t.Errorf("%d: %v", n, err)
		}
	}()

	data := make([]byte, 1024)
	if n, err := io.ReadAtLeast(client, data, 6); n != 7 || err != nil {
		t.Fatalf("%d: %v", n, err)
	}
	if !reflect.DeepEqual(data[:7], append([]byte{1}, []byte("server")...)) {
		t.Errorf("unexpected client read: %v", data[:7])
	}

	// verify that an ignore channel is empty in both directions.
	if n, err := conn.channels[2].Write([]byte("test")); n != 4 || err != nil {
		t.Errorf("writes should be ignored")
	}
	data = make([]byte, 1024)
	if n, err := conn.channels[2].Read(data); n != 0 || err != io.EOF {
		t.Errorf("reads should be ignored")
	}

	// verify that a write to a Read channel doesn't block
	if n, err := conn.channels[3].Write([]byte("test")); n != 4 || err != nil {
		t.Errorf("writes should be ignored")
	}

	// verify that a read from a Write channel doesn't block
	data = make([]byte, 1024)
	if n, err := conn.channels[4].Read(data); n != 0 || err != io.EOF {
		t.Errorf("reads should be ignored")
	}

	// verify that a client write to a Write channel doesn't block (is dropped)
	if n, err := client.Write(append([]byte{4}, []byte("ignored")...)); err != nil || n != 8 {
		t.Fatalf("%d: %v", n, err)
	}

	client.Close()
	wg.Wait()
}

func TestBase64Conn(t *testing.T) {
	conn := NewConn(NewDefaultChannelProtocols([]ChannelType{ReadWriteChannel, ReadWriteChannel}))
	s, addr := newServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conn.Open(w, req)
	}))
	defer s.Close()

	config, err := websocket.NewConfig("ws://"+addr, "http://localhost/")
	if err != nil {
		t.Fatal(err)
	}
	config.Protocol = []string{"base64.channel.k8s.io"}
	client, err := websocket.DialConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	<-conn.ready
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		data, err := io.ReadAll(conn.channels[0])
		if err != nil {
			t.Error(err)
			return
		}
		if !reflect.DeepEqual(data, []byte("client")) {
			t.Errorf("unexpected server read: %s", string(data))
		}
	}()

	clientData := base64.StdEncoding.EncodeToString([]byte("client"))
	if n, err := client.Write(append([]byte{'0'}, clientData...)); err != nil || n != len(clientData)+1 {
		t.Fatalf("%d: %v", n, err)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		if n, err := conn.channels[1].Write([]byte("server")); err != nil && n != 6 {
			t.Errorf("%d: %v", n, err)
		}
	}()

	data := make([]byte, 1024)
	if n, err := io.ReadAtLeast(client, data, 9); n != 9 || err != nil {
		t.Fatalf("%d: %v", n, err)
	}
	expect := []byte(base64.StdEncoding.EncodeToString([]byte("server")))

	if !reflect.DeepEqual(data[:9], append([]byte{'1'}, expect...)) {
		t.Errorf("unexpected client read: %v", data[:9])
	}

	client.Close()
	wg.Wait()
}

type versionTest struct {
	supported map[string]bool // protocol -> binary
	requested []string
	error     bool
	expected  string
}

func versionTests() []versionTest {
	const (
		binary = true
		base64 = false
	)
	return []versionTest{
		{
			supported: nil,
			requested: []string{"raw"},
			error:     true,
		},
		{
			supported: map[string]bool{"": binary, "raw": binary, "base64": base64},
			requested: nil,
			expected:  "",
		},
		{
			supported: map[string]bool{"": binary, "raw": binary, "base64": base64},
			requested: []string{"v1.raw"},
			error:     true,
		},
		{
			supported: map[string]bool{"": binary, "raw": binary, "base64": base64},
			requested: []string{"v1.raw", "v1.base64"},
			error:     true,
		}, {
			supported: map[string]bool{"": binary, "raw": binary, "base64": base64},
			requested: []string{"v1.raw", "raw"},
			expected:  "raw",
		},
		{
			supported: map[string]bool{"": binary, "v1.raw": binary, "v1.base64": base64, "v2.raw": binary, "v2.base64": base64},
			requested: []string{"v1.raw"},
			expected:  "v1.raw",
		},
		{
			supported: map[string]bool{"": binary, "v1.raw": binary, "v1.base64": base64, "v2.raw": binary, "v2.base64": base64},
			requested: []string{"v2.base64"},
			expected:  "v2.base64",
		},
	}
}

func TestVersionedConn(t *testing.T) {
	for i, test := range versionTests() {
		func() {
			supportedProtocols := map[string]ChannelProtocolConfig{}
			for p, binary := range test.supported {
				supportedProtocols[p] = ChannelProtocolConfig{
					Binary:   binary,
					Channels: []ChannelType{ReadWriteChannel},
				}
			}
			conn := NewConn(supportedProtocols)
			// note that it's not enough to wait for conn.ready to avoid a race here. Hence,
			// we use a channel.
			selectedProtocol := make(chan string)
			s, addr := newServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				p, _, _ := conn.Open(w, req)
				selectedProtocol <- p
			}))
			defer s.Close()

			config, err := websocket.NewConfig("ws://"+addr, "http://localhost/")
			if err != nil {
				t.Fatal(err)
			}
			config.Protocol = test.requested
			client, err := websocket.DialConfig(config)
			if err != nil {
				if !test.error {
					t.Fatalf("test %d: didn't expect error: %v", i, err)
				} else {
					return
				}
			}
			defer client.Close()
			if test.error && err == nil {
				t.Fatalf("test %d: expected an error", i)
			}

			<-conn.ready
			if got, expected := <-selectedProtocol, test.expected; got != expected {
				t.Fatalf("test %d: unexpected protocol version: got=%s expected=%s", i, got, expected)
			}
		}()
	}
}

func TestIsWebSocketRequestWithStreamCloseProtocol(t *testing.T) {
	tests := map[string]struct {
		headers  map[string]string
		expected bool
	}{
		"No headers returns false": {
			headers:  map[string]string{},
			expected: false,
		},
		"Only connection upgrade header is false": {
			headers: map[string]string{
				"Connection": "upgrade",
			},
			expected: false,
		},
		"Only websocket upgrade header is false": {
			headers: map[string]string{
				"Upgrade": "websocket",
			},
			expected: false,
		},
		"Only websocket and connection upgrade headers is false": {
			headers: map[string]string{
				"Connection": "upgrade",
				"Upgrade":    "websocket",
			},
			expected: false,
		},
		"Missing connection/upgrade header is false": {
			headers: map[string]string{
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v5.channel.k8s.io",
			},
			expected: false,
		},
		"Websocket connection upgrade headers with v5 protocol is true": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v5.channel.k8s.io",
			},
			expected: true,
		},
		"Websocket connection upgrade headers with wrong case v5 protocol is false": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v5.CHANNEL.k8s.io", // header value is case-sensitive
			},
			expected: false,
		},
		"Websocket connection upgrade headers with v4 protocol is false": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v4.channel.k8s.io",
			},
			expected: false,
		},
		"Websocket connection upgrade headers with multiple protocols but missing v5 is false": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v4.channel.k8s.io,v3.channel.k8s.io,v2.channel.k8s.io",
			},
			expected: false,
		},
		"Websocket connection upgrade headers with multiple protocols including v5 and spaces is true": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v5.channel.k8s.io,  v4.channel.k8s.io",
			},
			expected: true,
		},
		"Websocket connection upgrade headers with multiple protocols out of order including v5 and spaces is true": {
			headers: map[string]string{
				"Connection":            "upgrade",
				"Upgrade":               "websocket",
				WebSocketProtocolHeader: "v4.channel.k8s.io, v5.channel.k8s.io, v3.channel.k8s.io",
			},
			expected: true,
		},

		"Websocket connection upgrade headers key is case-insensitive": {
			headers: map[string]string{
				"Connection":             "upgrade",
				"Upgrade":                "websocket",
				"sec-websocket-protocol": "v4.channel.k8s.io, v5.channel.k8s.io, v3.channel.k8s.io",
			},
			expected: true,
		},
	}

	for name, test := range tests {
		req, err := http.NewRequest("GET", "http://www.example.com/", nil)
		require.NoError(t, err)
		for key, value := range test.headers {
			req.Header.Add(key, value)
		}
		actual := IsWebSocketRequestWithStreamCloseProtocol(req)
		assert.Equal(t, test.expected, actual, "%s: expected (%t), got (%t)", name, test.expected, actual)
	}
}

func TestProtocolSupportsStreamClose(t *testing.T) {
	tests := map[string]struct {
		protocol string
		expected bool
	}{
		"empty protocol returns false": {
			protocol: "",
			expected: false,
		},
		"not binary protocol returns false": {
			protocol: "base64.channel.k8s.io",
			expected: false,
		},
		"V1 protocol returns false": {
			protocol: "channel.k8s.io",
			expected: false,
		},
		"V4 protocol returns false": {
			protocol: "v4.channel.k8s.io",
			expected: false,
		},
		"V5 protocol returns true": {
			protocol: "v5.channel.k8s.io",
			expected: true,
		},
		"V5 protocol wrong case returns false": {
			protocol: "V5.channel.K8S.io",
			expected: false,
		},
	}

	for name, test := range tests {
		actual := protocolSupportsStreamClose(test.protocol)
		assert.Equal(t, test.expected, actual,
			"%s: expected (%t), got (%t)", name, test.expected, actual)
	}
}
