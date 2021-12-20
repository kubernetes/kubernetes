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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"k8s.io/klog/v2"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/util/exec"
)

// streamProtocolV4 implements version 4 of the streaming protocol for attach
// and exec.
type streamProtocolV4 struct {
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

	resizeTerminalIn  *io.PipeReader
	resizeTerminalOut *io.PipeWriter
}

var _ streamProtocolHandler = &streamProtocolV4{}

func newBinaryV4(options StreamOptions) streamProtocolHandler {
	return &streamProtocolV4{
		StreamOptions: options,
		binary:        true,
	}
}

func newBase64V4(options StreamOptions) streamProtocolHandler {
	return &streamProtocolV4{
		StreamOptions: options,
		binary:        false,
	}
}

func (p *streamProtocolV4) copyStdin() {
	if p.Stdin != nil {
		var once sync.Once

		// copy from client's stdin to container's stdin
		/*go func() {
			defer runtime.HandleCrash()

			// if p.stdin is noninteractive, p.g. `echo abc | kubectl exec -i <pod> -- cat`, make sure
			// we close remoteStdin as soon as the copy from p.stdin to remoteStdin finishes. Otherwise
			// the executed command will remain running.
			defer once.Do(func() { p.remoteStdinIn.Close() })

			if _, err := io.Copy(p.remoteStdinOut, readerWrapper{p.Stdin}); err != nil {
				runtime.HandleError(err)
			}
		}()*/

		// read from remoteStdin until the stream is closed. this is essential to
		// be able to exit interactive sessions cleanly and not leak goroutines or
		// hang the client's terminal.
		//
		// TODO we aren't using go-dockerclient any more; revisit this to determine if it's still
		// required by engine-api.
		//
		// go-dockerclient's current hijack implementation
		// (https://github.com/fsouza/go-dockerclient/blob/89f3d56d93788dfe85f864a44f85d9738fca0670/client.go#L564)
		// waits for all three streams (stdin/stdout/stderr) to finish copying
		// before returning. When hijack finishes copying stdout/stderr, it calls
		// Close() on its side of remoteStdin, which allows this copy to complete.
		// When that happens, we must Close() on our side of remoteStdin, to
		// allow the copy in hijack to complete, and hijack to return.
		go func() {
			defer runtime.HandleCrash()
			defer once.Do(func() { p.remoteStdinIn.Close() })

			// this "copy" doesn't actually read anything - it's just here to wait for
			// the server to close remoteStdin.
			if _, err := io.Copy(ioutil.Discard, p.remoteStdinIn); err != nil {
				p.closeStreams()
				runtime.HandleError(err)
			}
		}()
	}
}

func (p *streamProtocolV4) copyStdout(wg *sync.WaitGroup) {
	if p.Stdout == nil {
		return
	}

	wg.Add(1)
	go func() {
		defer runtime.HandleCrash()
		defer wg.Done()
		defer p.closeStreams()

		if _, err := io.Copy(p.Stdout, p.remoteStdoutIn); err != nil {
			if err != io.EOF {
				runtime.HandleError(err)
			}
		}
	}()
}

func (p *streamProtocolV4) copyStderr(wg *sync.WaitGroup) {
	if p.Stderr == nil || p.Tty {
		return
	}

	wg.Add(1)
	go func() {
		defer runtime.HandleCrash()
		defer wg.Done()
		defer p.closeStreams()

		if _, err := io.Copy(p.Stderr, p.remoteStderrIn); err != nil {
			runtime.HandleError(err)
		}
	}()
}

func (p *streamProtocolV4) stream(conn *websocket.Conn) error {

	defer conn.Close()
	doneChan := make(chan struct{}, 2)

	// set up error stream
	p.errorStreamIn, p.errorStreamOut = io.Pipe()

	// set up stdin stream
	if p.Stdin != nil {
		p.remoteStdinIn, p.remoteStdinOut = io.Pipe()
	}

	// set up stdout stream
	if p.Stdout != nil {
		p.remoteStdoutIn, p.remoteStdoutOut = io.Pipe()
	}

	// set up stderr stream
	if p.Stderr != nil && !p.Tty {
		p.remoteStderrIn, p.remoteStderrOut = io.Pipe()
	}

	// set up resize stream
	if p.Tty {
		p.resizeTerminalIn, p.resizeTerminalOut = io.Pipe()
	}

	// now that all the streams have been created, proceed with reading & copying

	errorChan := watchErrorStream(p.errorStreamIn, &errorDecoderV4{})

	var wg sync.WaitGroup

	//start streaming to the api server
	p.pullFromWebSocket(conn, &wg)

	//stream standard in
	p.pushToWebSocket(conn, &wg, p.Stdin, StreamStdIn, Base64StreamStdIn)

	//stream the resize stream

	if p.Tty {
		p.pushToWebSocket(conn, &wg, pipeReaderWrapper{reader: p.resizeTerminalIn}, StreamResize, Base64StreamResize)
	}

	p.copyStdout(&wg)
	p.copyStderr(&wg)

	p.handleResizes()

	p.copyStdin()
	//p.ping(conn, doneChan)

	// we're waiting for stdout/stderr to finish copying
	wg.Wait()

	// waits for errorStream to finish reading with an error or nil

	// notify the ping function to stop
	doneChan <- struct{}{}

	return <-errorChan
}

func (p *streamProtocolV4) closeStreams() {

	//close out the pipes

	//p.errorStreamIn.CloseWithError(nil)
	p.errorStreamOut.CloseWithError(nil)

	if p.remoteStderrIn != nil {
		//p.remoteStderrIn.CloseWithError(nil)
		p.remoteStderrOut.CloseWithError(nil)

	}

	if p.remoteStdinIn != nil {
		//p.remoteStdinIn.CloseWithError(nil)
		p.remoteStdinOut.CloseWithError(nil)

	}

	if p.remoteStdoutIn != nil {
		//p.remoteStdoutIn.CloseWithError(nil)
		p.remoteStdoutOut.CloseWithError(nil)

	}

	if p.resizeTerminalIn != nil {
		//p.resizeTerminalIn.CloseWithError(nil)
		p.resizeTerminalOut.CloseWithError(nil)

	}

}

func (p *streamProtocolV4) handleResizes() {
	if p.resizeTerminalOut == nil || p.TerminalSizeQueue == nil {
		return
	}
	go func() {
		defer runtime.HandleCrash()

		encoder := json.NewEncoder(p.resizeTerminalOut)
		for {
			size := p.TerminalSizeQueue.Next()
			if size == nil {
				return
			}
			if err := encoder.Encode(&size); err != nil {
				runtime.HandleError(err)
			}
		}
	}()
}

// errorDecoderV4 interprets the json-marshaled metav1.Status on the error channel
// and creates an exec.ExitError from it.
type errorDecoderV4 struct{}

func (d *errorDecoderV4) decode(message []byte) error {
	status := metav1.Status{}
	err := json.Unmarshal(message, &status)
	if err != nil {
		return fmt.Errorf("error stream protocol error: %v in %q", err, string(message))
	}
	switch status.Status {
	case metav1.StatusSuccess:
		return nil
	case metav1.StatusFailure:
		if status.Reason == remotecommand.NonZeroExitCodeReason {
			if status.Details == nil {
				return errors.New("error stream protocol error: details must be set")
			}
			for i := range status.Details.Causes {
				c := &status.Details.Causes[i]
				if c.Type != remotecommand.ExitCodeCauseType {
					continue
				}

				rc, err := strconv.ParseUint(c.Message, 10, 8)
				if err != nil {
					return fmt.Errorf("error stream protocol error: invalid exit code value %q", c.Message)
				}
				return exec.CodeExitError{
					Err:  fmt.Errorf("command terminated with exit code %d", rc),
					Code: int(rc),
				}
			}

			return fmt.Errorf("error stream protocol error: no %s cause given", remotecommand.ExitCodeCauseType)
		}
	default:
		return errors.New("error stream protocol error: unknown error")
	}

	return fmt.Errorf(status.Message)
}

func (p *streamProtocolV4) pushToWebSocket(conn *websocket.Conn, wg *sync.WaitGroup, in io.Reader, binaryChannelID byte, base64ChannelID byte) {
	if in == nil {
		return
	}
	wg.Add(1)

	origCloseHandler := conn.CloseHandler()
	conn.SetCloseHandler(func(code int, text string) error {
		wg.Done()
		if origCloseHandler != nil {
			return origCloseHandler(code, text)
		}
		return nil
	})

	go func() {
		defer runtime.HandleCrash()
		defer p.closeStreams()

		buffer := make([]byte, 1024)

		for {
			numberOfBytesRead, err := in.Read(buffer)

			if err != nil {
				if err == io.EOF {
					return
				} else {
					runtime.HandleError(err)
					err := conn.Close()
					if err != nil {
						klog.V(2).Infof("error in close websocket conn. %+v", err)
					}
					return
				}
			}

			var data []byte

			if p.binary {
				data = make([]byte, numberOfBytesRead+1)
				data = append([]byte{binaryChannelID}, buffer[0:numberOfBytesRead]...)
			} else {
				enc := base64.StdEncoding.EncodeToString(buffer[0:numberOfBytesRead])
				data = append([]byte{base64ChannelID}, []byte(enc)...)
			}

			err = conn.WriteMessage(websocket.BinaryMessage, data)
			if err != nil {
				runtime.HandleError(err)
				return
			}
		}
	}()
}

func (p *streamProtocolV4) pullFromWebSocket(conn *websocket.Conn, wg *sync.WaitGroup) {
	wg.Add(1)

	origCloseHandler := conn.CloseHandler()
	conn.SetCloseHandler(func(code int, text string) error {
		wg.Done()
		if origCloseHandler != nil {
			return origCloseHandler(code, text)
		}
		return nil
	})

	go func() {
		defer runtime.HandleCrash()
		defer p.closeStreams()
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
					if websocket.IsUnexpectedCloseError(websocketErr, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
						err := conn.Close()
						if err != nil {
							klog.V(2).Infof("error in close websocket conn. %+v", err)
							return
						}
						return
					}
					runtime.HandleError(err)
				} else {
					runtime.HandleError(err)
				}
				return
			}
		}
	}()
}

func (p *streamProtocolV4) ping(ws *websocket.Conn, done chan struct{}) {
	go func() {
		ticker := time.NewTicker(pingPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := ws.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(writeWait)); err != nil {
					runtime.HandleError(err)
				}
			case <-done:
				return
			}
		}
	}()
}
