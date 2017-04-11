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

package remotecommand

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"sync"
	"time"

	"bufio"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/runtime"
)

// streamProtocolV4 implements version 4 of the streaming protocol for attach
// and exec. This version adds support for exit codes on the error stream through
// the use of metav1.Status instead of plain text messages.
type streamProtocolV4 struct {
	*streamProtocolV3

	keepAliveChan chan int
}

var _ streamProtocolHandler = &streamProtocolV4{}

func newStreamProtocolV4(options StreamOptions) streamProtocolHandler {
	return &streamProtocolV4{
		streamProtocolV3: newStreamProtocolV3(options).(*streamProtocolV3),
		keepAliveChan:    make(chan int),
	}
}

func (p *streamProtocolV4) createStreams(conn streamCreator) error {
	return p.streamProtocolV3.createStreams(conn)
}

func (p *streamProtocolV4) handleResizes() {
	p.streamProtocolV3.handleResizes()
}

// copy bytes from p.remoteStdout to p.Stdout without using io.Copy, in order to
// reuse the stream to notify p.keepAliveChan of ongoing activity in the remote shell
func (p *streamProtocolV4) copyStdout(wg *sync.WaitGroup) {
	if p.Stdout == nil {
		return
	}

	wg.Add(1)
	go func() {
		defer runtime.HandleCrash()
		defer wg.Done()
		defer close(p.keepAliveChan)

		chunkReader := bufio.NewReader(p.remoteStdout)
		buffer := make([]byte, 32*1024)
		for {
			n, err := chunkReader.Read(buffer)
			if err == io.EOF {
				break
			}
			if err != nil {
				runtime.HandleError(err)
				break
			}

			p.Stdout.Write(buffer[0:n])

			// send amount of bytes written to keepAliveChan
			// in order to notify it of ongoing activity in
			// the remote shell.
			p.keepAliveChan <- n
		}
	}()
}

func (p *streamProtocolV4) stream(conn streamCreator) error {
	if err := p.createStreams(conn); err != nil {
		return err
	}

	// now that all the streams have been created, proceed with reading & copying

	errorChan := watchErrorStream(p.errorStream, &errorDecoderV4{})
	waitChan := make(chan bool)

	go func() {
		defer runtime.HandleCrash()

		// now that all the streams have been created, proceed with reading & copying

		p.handleResizes()
		p.copyStdin()

		var wg sync.WaitGroup
		p.copyStdout(&wg)
		p.copyStderr(&wg)

		// we're waiting for stdout/stderr to finish copying
		wg.Wait()

		close(waitChan)
	}()

	if p.StreamTimeout == 0 {
		for {
			select {
			case <-waitChan:
				return <-errorChan
			case <-p.keepAliveChan:
				// noop - keep reading from this channel to prevent
				// blocking the stdout stream copy.
			}
		}
	} else {
		totalIdleTime := p.StreamTimeout.Seconds()
		tickDuration := 1 * time.Second

		// if tickDuration is >= user-specified idle timeout,
		// make the ticker count equal to the user-defined timeout
		if p.StreamTimeout < tickDuration {
			tickDuration = p.StreamTimeout
		}

		for {
			select {
			case <-waitChan:
				return <-errorChan
			case <-p.keepAliveChan:
				totalIdleTime = p.StreamTimeout.Seconds()
			case <-time.After(tickDuration):
				totalIdleTime--
				if totalIdleTime <= 0 {
					return fmt.Errorf("Timeout exceeded for this operation.")
				}
			}
		}
	}
	return nil
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
