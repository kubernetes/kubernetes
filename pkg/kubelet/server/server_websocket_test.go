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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
)

const (
	dataChannel = iota
	errorChannel
)

func TestServeWSPortForward(t *testing.T) {
	tests := map[string]struct {
		port          string
		uid           bool
		clientData    string
		containerData string
		shouldError   bool
	}{
		"no port":                       {port: "", shouldError: true},
		"none number port":              {port: "abc", shouldError: true},
		"negative port":                 {port: "-1", shouldError: true},
		"too large port":                {port: "65536", shouldError: true},
		"0 port":                        {port: "0", shouldError: true},
		"min port":                      {port: "1", shouldError: false},
		"normal port":                   {port: "8000", shouldError: false},
		"normal port with data forward": {port: "8000", clientData: "client data", containerData: "container data", shouldError: false},
		"max port":                      {port: "65535", shouldError: false},
		"normal port with uid":          {port: "8000", uid: true, shouldError: false},
	}

	podNamespace := "other"
	podName := "foo"

	for desc := range tests {
		test := tests[desc]
		t.Run(desc, func(t *testing.T) {
			ss, err := newTestStreamingServer(0)
			require.NoError(t, err)
			defer ss.testHTTPServer.Close()
			fw := newServerTestWithDebug(true, false, ss)
			defer fw.testHTTPServer.Close()

			portForwardFuncDone := make(chan struct{})

			fw.fakeKubelet.getPortForwardCheck = func(name, namespace string, uid types.UID, opts portforward.V4Options) {
				assert.Equal(t, podName, name, "pod name")
				assert.Equal(t, podNamespace, namespace, "pod namespace")
				if test.uid {
					assert.Equal(t, testUID, string(uid), "uid")
				}
			}

			ss.fakeRuntime.portForwardFunc = func(podSandboxID string, port int32, stream io.ReadWriteCloser) error {
				defer close(portForwardFuncDone)
				assert.Equal(t, testPodSandboxID, podSandboxID, "pod sandbox id")
				// The port should be valid if it reaches here.
				testPort, err := strconv.ParseInt(test.port, 10, 32)
				require.NoError(t, err, "parse port")
				assert.Equal(t, int32(testPort), port, "port")

				if test.clientData != "" {
					fromClient := make([]byte, 32)
					n, err := stream.Read(fromClient)
					assert.NoError(t, err, "reading client data")
					assert.Equal(t, test.clientData, string(fromClient[0:n]), "client data")
				}

				if test.containerData != "" {
					_, err := stream.Write([]byte(test.containerData))
					assert.NoError(t, err, "writing container data")
				}

				return nil
			}

			var url string
			if test.uid {
				url = fmt.Sprintf("ws://%s/portForward/%s/%s/%s?port=%s", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName, testUID, test.port)
			} else {
				url = fmt.Sprintf("ws://%s/portForward/%s/%s?port=%s", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName, test.port)
			}

			ws, err := websocket.Dial(url, "", "http://127.0.0.1/")
			assert.Equal(t, test.shouldError, err != nil, "websocket dial")
			if test.shouldError {
				return
			}
			defer ws.Close()

			p, err := strconv.ParseUint(test.port, 10, 16)
			require.NoError(t, err, "parse port")
			p16 := uint16(p)

			channel, data, err := wsRead(ws)
			require.NoError(t, err, "read")
			assert.Equal(t, dataChannel, int(channel), "channel")
			assert.Len(t, data, binary.Size(p16), "data size")
			assert.Equal(t, p16, binary.LittleEndian.Uint16(data), "data")

			channel, data, err = wsRead(ws)
			assert.NoError(t, err, "read")
			assert.Equal(t, errorChannel, int(channel), "channel")
			assert.Len(t, data, binary.Size(p16), "data size")
			assert.Equal(t, p16, binary.LittleEndian.Uint16(data), "data")

			if test.clientData != "" {
				println("writing the client data")
				err := wsWrite(ws, dataChannel, []byte(test.clientData))
				assert.NoError(t, err, "writing client data")
			}

			if test.containerData != "" {
				_, data, err = wsRead(ws)
				assert.NoError(t, err, "reading container data")
				assert.Equal(t, test.containerData, string(data), "container data")
			}

			<-portForwardFuncDone
		})
	}
}

func TestServeWSMultiplePortForward(t *testing.T) {
	portsText := []string{"7000,8000", "9000"}
	ports := []uint16{7000, 8000, 9000}
	podNamespace := "other"
	podName := "foo"

	ss, err := newTestStreamingServer(0)
	require.NoError(t, err)
	defer ss.testHTTPServer.Close()
	fw := newServerTestWithDebug(true, false, ss)
	defer fw.testHTTPServer.Close()

	portForwardWG := sync.WaitGroup{}
	portForwardWG.Add(len(ports))

	portsMutex := sync.Mutex{}
	portsForwarded := map[int32]struct{}{}

	fw.fakeKubelet.getPortForwardCheck = func(name, namespace string, uid types.UID, opts portforward.V4Options) {
		assert.Equal(t, podName, name, "pod name")
		assert.Equal(t, podNamespace, namespace, "pod namespace")
	}

	ss.fakeRuntime.portForwardFunc = func(podSandboxID string, port int32, stream io.ReadWriteCloser) error {
		defer portForwardWG.Done()
		assert.Equal(t, testPodSandboxID, podSandboxID, "pod sandbox id")

		portsMutex.Lock()
		portsForwarded[port] = struct{}{}
		portsMutex.Unlock()

		fromClient := make([]byte, 32)
		n, err := stream.Read(fromClient)
		assert.NoError(t, err, "reading client data")
		assert.Equal(t, fmt.Sprintf("client data on port %d", port), string(fromClient[0:n]), "client data")

		_, err = stream.Write([]byte(fmt.Sprintf("container data on port %d", port)))
		assert.NoError(t, err, "writing container data")

		return nil
	}

	url := fmt.Sprintf("ws://%s/portForward/%s/%s?", fw.testHTTPServer.Listener.Addr().String(), podNamespace, podName)
	for _, port := range portsText {
		url = url + fmt.Sprintf("port=%s&", port)
	}

	ws, err := websocket.Dial(url, "", "http://127.0.0.1/")
	require.NoError(t, err, "websocket dial")

	defer ws.Close()

	for i, port := range ports {
		channel, data, err := wsRead(ws)
		assert.NoError(t, err, "port %d read", port)
		assert.Equal(t, i*2+dataChannel, int(channel), "port %d channel", port)
		assert.Len(t, data, binary.Size(port), "port %d data size", port)
		assert.Equal(t, binary.LittleEndian.Uint16(data), port, "port %d data", port)

		channel, data, err = wsRead(ws)
		assert.NoError(t, err, "port %d read", port)
		assert.Equal(t, i*2+errorChannel, int(channel), "port %d channel", port)
		assert.Len(t, data, binary.Size(port), "port %d data size", port)
		assert.Equal(t, binary.LittleEndian.Uint16(data), port, "port %d data", port)
	}

	for i, port := range ports {
		t.Logf("port %d writing the client data", port)
		err := wsWrite(ws, byte(i*2+dataChannel), []byte(fmt.Sprintf("client data on port %d", port)))
		assert.NoError(t, err, "port %d write client data", port)

		channel, data, err := wsRead(ws)
		assert.NoError(t, err, "port %d read container data", port)
		assert.Equal(t, i*2+dataChannel, int(channel), "port %d channel", port)
		assert.Equal(t, fmt.Sprintf("container data on port %d", port), string(data), "port %d container data", port)
	}

	portForwardWG.Wait()

	portsMutex.Lock()
	defer portsMutex.Unlock()
	assert.Len(t, portsForwarded, len(ports), "all ports forwarded")
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
