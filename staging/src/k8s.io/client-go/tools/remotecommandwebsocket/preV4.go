/*
Copyright 2020 The Kubernetes Authors.

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

package remotecommandwebsocket

import (
	"fmt"
	"io"
	"io/ioutil"
	"time"

	"encoding/base64"

	"github.com/gorilla/websocket"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
)

const ()

// preV4Protocol implements the first version of the streaming exec & attach
// protocol. This version has some bugs, such as not being able to detect when
// non-interactive stdin data has ended. See http://issues.k8s.io/13394 and
// http://issues.k8s.io/13395 for more details.
type preV4Protocol struct {
	StreamOptions

	binary bool

	errorStreamIn  *io.PipeReader
	errorStreamOut *io.PipeWriter

	remoteStdinIn  *io.PipeReader
	remoteStdinOut *io.PipeWriter

	remoteStdoutIn  *io.PipeReader
	remoteStdoutOut *io.PipeWriter

	remoteStderrIn  *io.PipeReader
	remoteStderrOut *io.PipeWriter
}

var _ streamProtocolHandler = &preV4Protocol{}

func newPreV4BinaryProtocol(options StreamOptions) streamProtocolHandler {
	return &preV4Protocol{
		StreamOptions: options,
		binary:        true,
	}
}

func newPreV4Base64Protocol(options StreamOptions) streamProtocolHandler {
	return &preV4Protocol{
		StreamOptions: options,
		binary:        false,
	}
}

func (p *preV4Protocol) stream(conn *websocket.Conn) error {
	defer conn.Close()
	doneChan := make(chan struct{}, 2)
	errorChan := make(chan error)

	cp := func(s string, dst io.Writer, src io.Reader) {
		klog.V(6).Infof("Copying %s", s)
		defer klog.V(6).Infof("Done copying %s", s)
		if _, err := io.Copy(dst, src); err != nil && err != io.EOF {
			klog.Errorf("Error copying %s: %v", s, err)
		}
		if s == v1.StreamTypeStdout || s == v1.StreamTypeStderr {
			doneChan <- struct{}{}
		}
	}

	// set up all the streams first
	p.errorStreamIn, p.errorStreamOut = io.Pipe()

	//defer p.errorStreamIn.

	// Create all the streams first, then start the copy goroutines. The server doesn't start its copy
	// goroutines until it's received all of the streams. If the client creates the stdin stream and
	// immediately begins copying stdin data to the server, it's possible to overwhelm and wedge the
	// spdy frame handler in the server so that it is full of unprocessed frames. The frames aren't
	// getting processed because the server hasn't started its copying, and it won't do that until it
	// gets all the streams. By creating all the streams first, we ensure that the server is ready to
	// process data before the client starts sending any. See https://issues.k8s.io/16373 for more info.
	if p.Stdin != nil {
		p.remoteStdinIn, p.remoteStdinOut = io.Pipe()
	}

	if p.Stdout != nil {
		p.remoteStdoutIn, p.remoteStdoutOut = io.Pipe()
	}

	if p.Stderr != nil && !p.Tty {
		p.remoteStderrIn, p.remoteStderrOut = io.Pipe()
	}

	// now that all the streams have been created, proceed with reading & copying
	// always read from errorStream
	go func() {
		message, err := ioutil.ReadAll(p.errorStreamIn)
		if err != nil && err != io.EOF {
			errorChan <- fmt.Errorf("Error reading from error stream: %s", err)
			return
		}
		if len(message) > 0 {
			errorChan <- fmt.Errorf("Error executing remote command: %s", message)
			return
		}
	}()

	if p.Stdin != nil {
		// TODO this goroutine will never exit cleanly (the io.Copy never unblocks)
		// because stdin is not closed until the process exits. If we try to call
		// stdin.Close(), it returns no error but doesn't unblock the copy. It will
		// exit when the process exits, instead.
		go cp(v1.StreamTypeStdin, p.remoteStdinOut, readerWrapper{p.remoteStdinIn})
	}

	waitCount := 0
	completedStreams := 0

	if p.Stdout != nil {
		waitCount++
		go cp(v1.StreamTypeStdout, p.Stdout, p.remoteStdoutIn)
	}

	if p.Stderr != nil && !p.Tty {
		waitCount++
		go cp(v1.StreamTypeStderr, p.Stderr, p.remoteStderrIn)
	}

	// setup pings to keep the connection alive during long operations
	go p.ping(conn, doneChan)
	waitCount++

	// pipes are connected, begin handling messages
	go p.pullFromWebSocket(conn, doneChan)
	if p.Stdin != nil {
		go p.pushToWebSocket(conn, doneChan)
	}

Loop:
	for {
		select {
		case <-doneChan:
			completedStreams++
			if completedStreams == waitCount {
				break Loop
			}
		case err := <-errorChan:
			return err
		}
	}

	return nil
}

func (p *preV4Protocol) pushToWebSocket(conn *websocket.Conn, doneChan chan struct{}) {
	buffer := make([]byte, 1024)

	for {

		numberOfBytesRead, err := p.StreamOptions.Stdin.Read(buffer)
		if err != nil {
			if err == io.EOF {
				doneChan <- struct{}{}
			} else {
				panic(err)
			}
		}

		var data []byte

		if p.binary {
			data = make([]byte, numberOfBytesRead+1)
			copy(data[1:], buffer[:])
			data[0] = StreamStdIn
		} else {
			enc := base64.StdEncoding.EncodeToString(buffer[0:numberOfBytesRead])
			data = append([]byte{'0'}, []byte(enc)...)
		}

		err = conn.WriteMessage(websocket.BinaryMessage, data)
		if err != nil {
			panic(err)
		}

	}

}

func (p *preV4Protocol) pullFromWebSocket(conn *websocket.Conn, doneChan chan struct{}) {
	defer runtime.HandleCrash()
	conn.SetReadLimit(maxMessageSize)
	conn.SetReadDeadline(time.Now().Add(pongWait))
	conn.SetPongHandler(func(string) error { conn.SetReadDeadline(time.Now().Add(pongWait)); return nil })
	buffer := make([]byte, 1024)
	for {
		messageType, message, err := conn.ReadMessage()

		if messageType > 0 {

			if p.binary {
				if len(message) > 0 {
					switch message[0] {
					case StreamStdOut:

						if _, err := p.remoteStdoutOut.Write(message[1:]); err != nil {
							runtime.HandleError(err)
						}
					case StreamStdErr:
						if _, err := p.remoteStderrOut.Write(message[1:]); err != nil {
							runtime.HandleError(err)
						}
					case StreamErr:
						if _, err := p.errorStreamOut.Write(message[1:]); err != nil {
							runtime.HandleError(err)
						}
					}
				}
			} else {
				if len(message) > 0 {
					numBytes, err := base64.StdEncoding.Decode(buffer, message[1:])
					if err != nil {
						runtime.HandleError(err)
					}

					switch message[0] {
					case Base64StreamStdOut:

						//fmt.Println(buffer)
						if _, err := p.remoteStdoutOut.Write(buffer[1:numBytes]); err != nil {
							runtime.HandleError(err)
						}
					case Base64StreamStdErr:
						if _, err := p.remoteStderrOut.Write(buffer[1:numBytes]); err != nil {
							runtime.HandleError(err)
						}
					case Base64StreamErr:
						if _, err := p.errorStreamOut.Write(buffer[1:numBytes]); err != nil {
							runtime.HandleError(err)
						}
					}
				}
			}

		}

		if err != nil {
			websocketErr, ok := err.(*websocket.CloseError)
			if ok {
				if websocketErr.Code == WebSocketExitStream {
					doneChan <- struct{}{}
				} else {
					runtime.HandleError(err)
				}
			} else {
				runtime.HandleError(err)
			}
		}
	}
}

func (p *preV4Protocol) ping(ws *websocket.Conn, done chan struct{}) {
	ticker := time.NewTicker(pingPeriod)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if err := ws.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(writeWait)); err != nil {
				panic(err)
			}
		case <-done:
			return
		}
	}
}
