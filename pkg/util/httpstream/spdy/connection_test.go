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

package spdy

import (
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/httpstream"
)

func runProxy(t *testing.T, backendUrl string, proxyUrl chan<- string, proxyDone chan<- struct{}) {
	listener, err := net.Listen("tcp4", "localhost:0")
	if err != nil {
		t.Fatalf("error listening: %v", err)
	}
	defer listener.Close()

	proxyUrl <- listener.Addr().String()

	clientConn, err := listener.Accept()
	if err != nil {
		t.Errorf("proxy: error accepting client connection: %v", err)
		return
	}

	backendConn, err := net.Dial("tcp4", backendUrl)
	if err != nil {
		t.Errorf("proxy: error dialing backend: %v", err)
		return
	}
	defer backendConn.Close()

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		io.Copy(backendConn, clientConn)
	}()

	go func() {
		defer wg.Done()
		io.Copy(clientConn, backendConn)
	}()

	wg.Wait()

	proxyDone <- struct{}{}
}

func runServer(t *testing.T, backendUrl chan<- string, serverDone chan<- struct{}) {
	listener, err := net.Listen("tcp4", "localhost:0")
	if err != nil {
		t.Fatalf("server: error listening: %v", err)
	}
	defer listener.Close()

	backendUrl <- listener.Addr().String()

	conn, err := listener.Accept()
	if err != nil {
		t.Errorf("server: error accepting connection: %v", err)
		return
	}

	streamChan := make(chan httpstream.Stream)
	replySentChan := make(chan (<-chan struct{}))
	spdyConn, err := NewServerConnection(conn, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamChan <- stream
		replySentChan <- replySent
		return nil
	})
	if err != nil {
		t.Errorf("server: error creating spdy connection: %v", err)
		return
	}

	stream := <-streamChan
	replySent := <-replySentChan
	<-replySent

	buf := make([]byte, 1)
	_, err = stream.Read(buf)
	if err != io.EOF {
		t.Errorf("server: unexpected read error: %v", err)
		return
	}

	<-spdyConn.CloseChan()
	raw := spdyConn.(*connection).conn
	if err := raw.Wait(15 * time.Second); err != nil {
		t.Errorf("server: timed out waiting for connection closure: %v", err)
	}

	serverDone <- struct{}{}
}

func TestConnectionCloseIsImmediateThroughAProxy(t *testing.T) {
	serverDone := make(chan struct{})
	backendUrlChan := make(chan string)
	go runServer(t, backendUrlChan, serverDone)
	backendUrl := <-backendUrlChan

	proxyDone := make(chan struct{})
	proxyUrlChan := make(chan string)
	go runProxy(t, backendUrl, proxyUrlChan, proxyDone)
	proxyUrl := <-proxyUrlChan

	conn, err := net.Dial("tcp4", proxyUrl)
	if err != nil {
		t.Fatalf("client: error connecting to proxy: %v", err)
	}

	spdyConn, err := NewClientConnection(conn)
	if err != nil {
		t.Fatalf("client: error creating spdy connection: %v", err)
	}

	if _, err := spdyConn.CreateStream(http.Header{}); err != nil {
		t.Fatalf("client: error creating stream: %v", err)
	}

	spdyConn.Close()
	raw := spdyConn.(*connection).conn
	if err := raw.Wait(15 * time.Second); err != nil {
		t.Fatalf("client: timed out waiting for connection closure: %v", err)
	}

	expired := time.NewTimer(15 * time.Second)
	defer expired.Stop()
	i := 0
	for {
		select {
		case <-expired.C:
			t.Fatalf("timed out waiting for proxy and/or server closure")
		case <-serverDone:
			i++
		case <-proxyDone:
			i++
		}
		if i == 2 {
			break
		}
	}
}
