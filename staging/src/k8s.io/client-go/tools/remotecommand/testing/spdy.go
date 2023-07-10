/*
Copyright 2023 The Kubernetes Authors.

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

package testing

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/remotecommand"
)

type Options struct {
	Stdin  bool
	Stdout bool
	Stderr bool
}

type StreamContext struct {
	Conn         io.Closer
	StdinStream  io.ReadCloser
	StdoutStream io.WriteCloser
	StderrStream io.WriteCloser
	WriteStatus  func(status *apierrors.StatusError) error
}

type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

func CreateHTTPStreams(w http.ResponseWriter, req *http.Request, opts Options) (*StreamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{remotecommand.StreamProtocolV4Name})
	if err != nil {
		return nil, err
	}

	upgrader := spdy.NewResponseUpgrader()
	streamCh := make(chan streamAndReply)
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	ctx := &StreamContext{
		Conn: conn,
	}

	// wait for stream
	replyChan := make(chan struct{}, 4)
	defer close(replyChan)
	receivedStreams := 0
	expectedStreams := 1
	if opts.Stdout {
		expectedStreams++
	}
	if opts.Stdin {
		expectedStreams++
	}
	if opts.Stderr {
		expectedStreams++
	}
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(v1.StreamType)
			switch streamType {
			case v1.StreamTypeError:
				replyChan <- struct{}{}
				ctx.WriteStatus = v4WriteStatusFunc(stream)
			case v1.StreamTypeStdout:
				replyChan <- struct{}{}
				ctx.StdoutStream = stream
			case v1.StreamTypeStdin:
				replyChan <- struct{}{}
				ctx.StdinStream = stream
			case v1.StreamTypeStderr:
				replyChan <- struct{}{}
				ctx.StderrStream = stream
			default:
				// add other stream ...
				return nil, errors.New("unimplemented stream type")
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		}
	}

	return ctx, nil
}

func v4WriteStatusFunc(stream io.Writer) func(status *apierrors.StatusError) error {
	return func(status *apierrors.StatusError) error {
		bs, err := json.Marshal(status.Status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}
