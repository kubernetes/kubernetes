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
	"io"
	"net/http"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
)

// streamProtocolV3 implements version 3 of the streaming protocol for attach
// and exec. This version adds support for resizing the container's terminal.
type streamProtocolV3 struct {
	*streamProtocolV2

	resizeStream io.Writer
}

var _ streamProtocolHandler = &streamProtocolV3{}

func newStreamProtocolV3(options StreamOptions) streamProtocolHandler {
	return &streamProtocolV3{
		streamProtocolV2: newStreamProtocolV2(options).(*streamProtocolV2),
	}
}

func (p *streamProtocolV3) createStreams(conn streamCreator) error {
	// set up the streams from v2
	if err := p.streamProtocolV2.createStreams(conn); err != nil {
		return err
	}

	// set up resize stream
	if p.Tty {
		headers := http.Header{}
		headers.Set(v1.StreamType, v1.StreamTypeResize)
		var err error
		p.resizeStream, err = conn.CreateStream(headers)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *streamProtocolV3) handleResizes() {
	if p.resizeStream == nil || p.TerminalSizeQueue == nil {
		return
	}
	go func() {
		defer runtime.HandleCrash()

		encoder := json.NewEncoder(p.resizeStream)
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

func (p *streamProtocolV3) stream(conn streamCreator) error {
	if err := p.createStreams(conn); err != nil {
		return err
	}

	// now that all the streams have been created, proceed with reading & copying

	errorChan := watchErrorStream(p.errorStream, &errorDecoderV3{})

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

type errorDecoderV3 struct {
	errorDecoderV2
}
