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
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/types"
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

		fw.fakeKubelet.portForwardFunc = func(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error {
			defer close(portForwardFuncDone)

			if e, a := expectedPodName, name; e != a {
				t.Fatalf("%d: pod name: expected '%v', got '%v'", i, e, a)
			}

			if e, a := expectedUid, uid; test.uid && e != string(a) {
				t.Fatalf("%d: uid: expected '%v', got '%v'", i, e, a)
			}

			p, err := strconv.ParseInt(test.port, 10, 32)
			if err != nil {
				t.Fatalf("%d: error parsing port string '%s': %v", i, test.port, err)
			}
			if e, a := int32(p), port; e != a {
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
		} else if err != nil {
			t.Fatalf("%d: websocket dial unexpected err: %v", i, err)
		}

		defer ws.Close()

		p, err := strconv.ParseUint(test.port, 10, 16)
		if err != nil {
			t.Fatalf("%d: error parsing port string '%s': %v", i, test.port, err)
		}
		p16 := uint16(p)

		channel, data, err := wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read failed: expected no error: got %v", i, err)
		}
		if channel != dataChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, dataChannel)
		}
		if len(data) != binary.Size(p16) {
			t.Fatalf("%d: wrong data size: got %q: expected %d", i, data, binary.Size(p16))
		}
		if e, a := p16, binary.LittleEndian.Uint16(data); e != a {
			t.Fatalf("%d: wrong data: got %q: expected %s", i, data, test.port)
		}

		channel, data, err = wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read succeeded: expected no error: got %v", i, err)
		}
		if channel != errorChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, errorChannel)
		}
		if len(data) != binary.Size(p16) {
			t.Fatalf("%d: wrong data size: got %q: expected %d", i, data, binary.Size(p16))
		}
		if e, a := p16, binary.LittleEndian.Uint16(data); e != a {
			t.Fatalf("%d: wrong data: got %q: expected %s", i, data, test.port)
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

func TestServeWSMultiplePortForward(t *testing.T) {
	portsText := []string{"7000,8000", "9000"}
	ports := []uint16{7000, 8000, 9000}
	podNamespace := "other"
	podName := "foo"
	expectedPodName := getPodName(podName, podNamespace)

	fw := newServerTest()
	defer fw.testHTTPServer.Close()

	fw.fakeKubelet.streamingConnectionIdleTimeoutFunc = func() time.Duration {
		return 0
	}

	portForwardWG := sync.WaitGroup{}
	portForwardWG.Add(len(ports))

	portsMutex := sync.Mutex{}
	portsForwarded := map[int32]struct{}{}

	fw.fakeKubelet.portForwardFunc = func(name string, uid types.UID, port int32, stream io.ReadWriteCloser) error {
		defer portForwardWG.Done()

		if e, a := expectedPodName, name; e != a {
			t.Fatalf("%d: pod name: expected '%v', got '%v'", port, e, a)
		}

		portsMutex.Lock()
		portsForwarded[port] = struct{}{}
		portsMutex.Unlock()

		fromClient := make([]byte, 32)
		n, err := stream.Read(fromClient)
		if err != nil {
			t.Fatalf("%d: error reading client data: %v", port, err)
		}
		if e, a := fmt.Sprintf("client data on port %d", port), string(fromClient[0:n]); e != a {
			t.Fatalf("%d: client data: expected to receive '%v', got '%v'", port, e, a)
		}

		_, err = stream.Write([]byte(fmt.Sprintf("container data on port %d", port)))
		if err != nil {
			t.Fatalf("%d: error writing container data: %v", port, err)
		}

		return nil
	}

	url := fmt.Sprintf("ws://%s/portForward/%s/%s?", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName)
	for _, port := range portsText {
		url = url + fmt.Sprintf("port=%s&", port)
	}

	ws, err := websocket.Dial(url, "", "http://127.0.0.1/")
	if err != nil {
		t.Fatalf("websocket dial unexpected err: %v", err)
	}

	defer ws.Close()

	for i, port := range ports {
		channel, data, err := wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read failed: expected no error: got %v", i, err)
		}
		if int(channel) != i*2+dataChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, i*2+dataChannel)
		}
		if len(data) != binary.Size(port) {
			t.Fatalf("%d: wrong data size: got %q: expected %d", i, data, binary.Size(port))
		}
		if e, a := port, binary.LittleEndian.Uint16(data); e != a {
			t.Fatalf("%d: wrong data: got %q: expected %d", i, data, port)
		}

		channel, data, err = wsRead(ws)
		if err != nil {
			t.Fatalf("%d: read succeeded: expected no error: got %v", i, err)
		}
		if int(channel) != i*2+errorChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", i, channel, i*2+errorChannel)
		}
		if len(data) != binary.Size(port) {
			t.Fatalf("%d: wrong data size: got %q: expected %d", i, data, binary.Size(port))
		}
		if e, a := port, binary.LittleEndian.Uint16(data); e != a {
			t.Fatalf("%d: wrong data: got %q: expected %d", i, data, port)
		}
	}

	for i, port := range ports {
		println("writing the client data", port)
		err := wsWrite(ws, byte(i*2+dataChannel), []byte(fmt.Sprintf("client data on port %d", port)))
		if err != nil {
			t.Fatalf("%d: unexpected error writing client data: %v", i, err)
		}

		channel, data, err := wsRead(ws)
		if err != nil {
			t.Fatalf("%d: unexpected error reading container data: %v", i, err)
		}

		if int(channel) != i*2+dataChannel {
			t.Fatalf("%d: wrong channel: got %q: expected %q", port, channel, i*2+dataChannel)
		}
		if e, a := fmt.Sprintf("container data on port %d", port), string(data); e != a {
			t.Fatalf("%d: expected to receive '%v' from container, got '%v'", i, e, a)
		}
	}

	portForwardWG.Wait()

	portsMutex.Lock()
	defer portsMutex.Unlock()
	if len(ports) != len(portsForwarded) {
		t.Fatalf("expected to forward %d ports; got %v", len(ports), portsForwarded)
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
