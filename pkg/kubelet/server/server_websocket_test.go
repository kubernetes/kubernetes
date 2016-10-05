/*
Copyright 2016 The Kubernetes Authors.

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

package server

import (
	"fmt"
	"io"
	"strconv"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/kubernetes/pkg/types"
)

const (
	dataChannel = iota
	errorChannel
)

func TestServeWSPortForward(t *testing.T) {
	tests := []struct {
		port          string
		uid           bool
		clientData    string
		containerData string
		shouldError   bool
	}{
		{port: "", shouldError: true},
		{port: "abc", shouldError: true},
		{port: "-1", shouldError: true},
		{port: "65536", shouldError: true},
		{port: "0", shouldError: true},
		{port: "1", shouldError: false},
		{port: "8000", shouldError: false},
		{port: "8000", clientData: "client data", containerData: "container data", shouldError: false},
		{port: "65535", shouldError: false},
		{port: "65535", uid: true, shouldError: false},
	}

	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)
	expectedUid := "9b01b80f-8fb4-11e4-95ab-4200af06647"

	for i, test := range tests {
		fw := newServerTest()
		defer fw.testHTTPServer.Close()

		fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
			return 0
		}

		portForwardFuncDone := make(chan struct{})

		fw.fakeKubelet.portForwardFunc = func(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error {
			defer close(portForwardFuncDone)

			if e, a := expectedPodName, name; e != a {
				t.Fatalf("%d: pod name: expected '%v', got '%v'", i, e, a)
			}

			if e, a := expectedUid, uid; test.uid && e != string(a) {
				t.Fatalf("%d: uid: expected '%v', got '%v'", i, e, a)
			}

			p, err := strconv.ParseUint(test.port, 10, 16)
			if err != nil {
				t.Fatalf("%d: error parsing port string '%s': %v", i, test.port, err)
			}
			if e, a := uint16(p), port; e != a {
				t.Fatalf("%d: port: expected '%v', got '%v'", i, e, a)
			}

			if test.clientData != "" {
				fromClient := make([]byte, 32)
				n, err := stream.Read(fromClient)
				if err != nil {
					t.Fatalf("%d: error reading client data: %v", i, err)
				}
				if e, a := test.clientData, string(fromClient[0:n]); e != a {
					t.Fatalf("%d: client data: expected to receive '%v', got '%v'", i, e, a)
				}
			}

			if test.containerData != "" {
				_, err := stream.Write([]byte(test.containerData))
				if err != nil {
					t.Fatalf("%d: error writing container data: %v", i, err)
				}
			}

			return nil
		}

		var url string
		if test.uid {
			url = fmt.Sprintf("ws://%s/portForward/%s/%s/%s?port=%s", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName, expectedUid, test.port)
		} else {
			url = fmt.Sprintf("ws://%s/portForward/%s/%s?port=%s", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName, test.port)
		}

		ws, err := websocket.Dial(url, "", "http://127.0.0.1/")
		if test.shouldError {
			if err == nil {
				t.Fatalf("%d: websocket dial expected err", i)
			}
			continue
		}
		defer ws.Close()

		channel, data, err := wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read failed: expected no error: got %v", i, err)
		}
		if channel != dataChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, dataChannel)
		}
		if len(data) != 0 {
			t.Fatalf("%d: wrong data: got %q: expected nothing", i, data)
		}

		channel, data, err = wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read succeeded: expected no error: got %v", i, err)
		}
		if channel != errorChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, errorChannel)
		}
		if len(data) != 0 {
			t.Fatalf("%d: wrong data: got %q: expected nothing", i, data)
		}

		if test.clientData != "" {
			println("writing the client data")
			err := wsWrite(ws, dataChannel, []byte(test.clientData))
			if err != nil {
				t.Fatalf("%d: unexpected error writing client data: %v", i, err)
			}
		}

		if test.containerData != "" {
			channel, data, err = wsRead(ws)
			if err != nil {
				t.Fatalf("%d: unexpected error reading container data: %v", i, err)
			}
			if e, a := test.containerData, string(data); e != a {
				t.Fatalf("%d: expected to receive '%v' from container, got '%v'", i, e, a)
			}
		}

		<-portForwardFuncDone
	}
}

func wsWrite(conn *websocket.Conn, channel byte, data []byte) error {
	frame := make([]byte, len(data)+1)
	frame[0] = channel
	copy(frame[1:], data)
	err := websocket.Message.Send(conn, frame)
	return err
}

func wsRead(conn *websocket.Conn) (byte, []byte, error) {
	for {
		var data []byte
		err := websocket.Message.Receive(conn, &data)
		if err != nil {
			return 0, nil, err
		}

		if len(data) == 0 {
			continue
		}

		channel := data[0]
		data = data[1:]

		return channel, data, err
	}
}
