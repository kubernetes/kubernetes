/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package remotecommand

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/runtime"
)

// streamProtocolV2 implements version 2 of the streaming protocol for attach
// and exec. The original streaming protocol was unversioned. As a result, this
// version is referred to as version 2, even though it is the first actual
// numbered version.
type streamProtocolV2 struct {
	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
	tty    bool
}

var _ streamProtocolHandler = &streamProtocolV2{}

func (e *streamProtocolV2) stream(conn httpstream.Connection) error {
	var (
		err                                                  error
		errorStream, remoteStdin, remoteStdout, remoteStderr httpstream.Stream
	)

	headers := http.Header{}

	// set up all the streams first
	// set up error stream
	errorChan := make(chan error)
	headers.Set(api.StreamType, api.StreamTypeError)
	errorStream, err = conn.CreateStream(headers)
	if err != nil {
		return err
	}

	// set up stdin stream
	if e.stdin != nil {
		headers.Set(api.StreamType, api.StreamTypeStdin)
		remoteStdin, err = conn.CreateStream(headers)
		if err != nil {
			return err
		}
	}

	// set up stdout stream
	if e.stdout != nil {
		headers.Set(api.StreamType, api.StreamTypeStdout)
		remoteStdout, err = conn.CreateStream(headers)
		if err != nil {
			return err
		}
	}

	// set up stderr stream
	if e.stderr != nil && !e.tty {
		headers.Set(api.StreamType, api.StreamTypeStderr)
		remoteStderr, err = conn.CreateStream(headers)
		if err != nil {
			return err
		}
	}

	// now that all the streams have been created, proceed with reading & copying

	// always read from errorStream
	go func() {
		message, err := ioutil.ReadAll(errorStream)
		switch {
		case err != nil && err != io.EOF:
			errorChan <- fmt.Errorf("error reading from error stream: %s", err)
		case len(message) > 0:
			errorChan <- fmt.Errorf("error executing remote command: %s", message)
		default:
			errorChan <- nil
		}
		close(errorChan)
	}()

	var wg sync.WaitGroup
	var once sync.Once

	if e.stdin != nil {
		// copy from client's stdin to container's stdin
		go func() {
			// if e.stdin is noninteractive, e.g. `echo abc | kubectl exec -i <pod> -- cat`, make sure
			// we close remoteStdin as soon as the copy from e.stdin to remoteStdin finishes. Otherwise
			// the executed command will remain running.
			defer once.Do(func() { remoteStdin.Close() })

			if _, err := io.Copy(remoteStdin, e.stdin); err != nil {
				runtime.HandleError(err)
			}
		}()

		// read from remoteStdin until the stream is closed. this is essential to
		// be able to exit interactive sessions cleanly and not leak goroutines or
		// hang the client's terminal.
		//
		// go-dockerclient's current hijack implementation
		// (https://github.com/fsouza/go-dockerclient/blob/89f3d56d93788dfe85f864a44f85d9738fca0670/client.go#L564)
		// waits for all three streams (stdin/stdout/stderr) to finish copying
		// before returning. When hijack finishes copying stdout/stderr, it calls
		// Close() on its side of remoteStdin, which allows this copy to complete.
		// When that happens, we must Close() on our side of remoteStdin, to
		// allow the copy in hijack to complete, and hijack to return.
		go func() {
			defer once.Do(func() { remoteStdin.Close() })
			// this "copy" doesn't actually read anything - it's just here to wait for
			// the server to close remoteStdin.
			if _, err := io.Copy(ioutil.Discard, remoteStdin); err != nil {
				runtime.HandleError(err)
			}
		}()
	}

	if e.stdout != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := io.Copy(e.stdout, remoteStdout); err != nil {
				runtime.HandleError(err)
			}
		}()
	}

	if e.stderr != nil && !e.tty {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := io.Copy(e.stderr, remoteStderr); err != nil {
				runtime.HandleError(err)
			}
		}()
	}

	// we're waiting for stdout/stderr to finish copying
	wg.Wait()

	// waits for errorStream to finish reading with an error or nil
	return <-errorChan
}
