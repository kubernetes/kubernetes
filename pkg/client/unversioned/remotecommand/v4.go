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
	"fmt"
	"sync"

	"errors"
	"k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/util/exec"
)

// streamProtocolV4 implements version 4 of the streaming protocol for attach
// and exec. This version adds support exit code on the error stream.
type streamProtocolV4 struct {
	*streamProtocolV3
}

var _ streamProtocolHandler = &streamProtocolV4{}

func newStreamProtocolV4(options StreamOptions) streamProtocolHandler {
	return &streamProtocolV4{
		streamProtocolV3: newStreamProtocolV3(options).(*streamProtocolV3),
	}
}

func (p *streamProtocolV4) createStreams(conn streamCreator) error {
	return p.streamProtocolV3.createStreams(conn)
}

func (p *streamProtocolV4) handleResizes() {
	p.streamProtocolV3.handleResizes()
}

func (p *streamProtocolV4) stream(conn streamCreator) error {
	if err := p.createStreams(conn); err != nil {
		return err
	}

	// now that all the streams have been created, proceed with reading & copying

	errorChan := watchErrorStream(p.errorStream, &errorDecoderV4{})

	p.handleResizes()

	p.copyStdin()

	var wg sync.WaitGroup
	p.copyStdout(&wg)
	p.copyStderr(&wg)

	// we're waiting for stdout/stderr to finish copying
	wg.Wait()

	// waits for errorStream to finish reading with an error or nil
	return <-errorChan
}

// errorDecoderV4 interprets the json data on the error channel and creates an uexec.ExitError from it.
type errorDecoderV4 struct{}

func (d *errorDecoderV4) decode(message []byte) error {
	m := remotecommand.V4ExitErrorMessage{}
	err := json.Unmarshal(message, &m)
	if err != nil {
		return fmt.Errorf("error stream protocol error: %v in %q", err, string(message))
	}
	if m.ExitCode == nil {
		return fmt.Errorf(m.Message)
	}
	return exec.CodeExitError{
		Err:  errors.New(m.Message),
		Code: *m.ExitCode,
	}
}
