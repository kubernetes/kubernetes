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

	"k8s.io/apimachinery/pkg/util/httpstream"
)

func runProxy(t *testing.T, backendUrl string, proxyUrl chan<- string, proxyDone chan<- struct{}, errCh chan<- error) {
	listener, err := net.Listen("tcp4", "localhost:0")
	if err != nil {
		errCh <- err
		return
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

func runServer(t *testing.T, backendUrl chan<- string, serverDone chan<- struct{}, errCh chan<- error) {
	listener, err := net.Listen("tcp4", "localhost:0")
	if err != nil {
		errCh <- err
		return
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
	errCh := make(chan error)

	serverDone := make(chan struct{}, 1)
	backendUrlChan := make(chan string)
	go runServer(t, backendUrlChan, serverDone, errCh)

	var backendUrl string
	select {
	case err := <-errCh:
		t.Fatalf("server: error listening: %v", err)
	case backendUrl = <-backendUrlChan:
	}

	proxyDone := make(chan struct{}, 1)
	proxyUrlChan := make(chan string)
	go runProxy(t, backendUrl, proxyUrlChan, proxyDone, errCh)

	var proxyUrl string
	select {
	case err := <-errCh:
		t.Fatalf("error listening: %v", err)
	case proxyUrl = <-proxyUrlChan:
	}

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

type fakeStream struct{ id uint32 }

func (*fakeStream) Read(p []byte) (int, error)  { return 0, nil }
func (*fakeStream) Write(p []byte) (int, error) { return 0, nil }
func (*fakeStream) Close() error                { return nil }
func (*fakeStream) Reset() error                { return nil }
func (*fakeStream) Headers() http.Header        { return nil }
func (f *fakeStream) Identifier() uint32        { return f.id }

func TestConnectionRemoveStreams(t *testing.T) {
	c := &connection{streams: make(map[uint32]httpstream.Stream)}
	stream0 := &fakeStream{id: 0}
	stream1 := &fakeStream{id: 1}
	stream2 := &fakeStream{id: 2}

	c.registerStream(stream0)
	c.registerStream(stream1)

	if len(c.streams) != 2 {
		t.Fatalf("should have two streams, has %d", len(c.streams))
	}

	// not exists
	c.RemoveStreams(stream2)

	if len(c.streams) != 2 {
		t.Fatalf("should have two streams, has %d", len(c.streams))
	}

	// remove all existing
	c.RemoveStreams(stream0, stream1)

	if len(c.streams) != 0 {
		t.Fatalf("should not have any streams, has %d", len(c.streams))
	}

}
