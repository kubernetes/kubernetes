/*
Copyright 2014 The Kubernetes Authors.

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

package bulk

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
)

type websocketSink struct {
	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder
	wsTimeout      time.Duration

	bc bulkConnection
	ws *websocket.Conn

	streamBuf bytes.Buffer // write buffer
	requests  chan *bulkapi.ClientMessage
}

func (s *websocketSink) resetTimeout() {
	if s.wsTimeout > 0 {
		if err := s.ws.SetDeadline(time.Now().Add(s.wsTimeout)); err != nil {
			utilruntime.HandleError(err)
		}
	}
}

func (s *websocketSink) abortConnection(err error) {
	if err != nil {
		utilruntime.HandleError(err)
	}
	s.ws.Close()
}

// Reads incoming requests from WS, validates them and run `handleRequest`
func (s *websocketSink) runReadAndProcessRequestsFromWSLoop(ctx context.Context, done <-chan struct{}) error {
	defer utilruntime.HandleCrash()
	defer close(s.requests)
	defaultGVK := s.bc.APIManager().GroupVersion.WithKind("ClientMessage")
	var data []byte
	for {
		s.resetTimeout()
		if err := websocket.Message.Receive(s.ws, &data); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("unable to receive message: %v", err)
		}
		if len(data) == 0 {
			continue
		}
		reqRaw, _, err := s.decoder.Decode(data, &defaultGVK, &bulkapi.ClientMessage{})
		if err != nil {
			return fmt.Errorf("unable to decode bulk request: %v", err)
		}
		req, ok := reqRaw.(*bulkapi.ClientMessage)
		if !ok {
			return fmt.Errorf("unable to decode bulk request: cast error")
		}

		select {
		case s.requests <- req:
			continue
		case <-ctx.Done():
			return ctx.Err()
		case <-done:
			return nil
		}
	}
}

func (s *websocketSink) WriteResponse(resp *bulkapi.ServerMessage) error {
	s.resetTimeout()
	if err := s.encoder.Encode(resp, &s.streamBuf); err != nil {
		return fmt.Errorf("unable to encode event: %v, %v", err, resp)
	}
	var data interface{}
	if s.serializerInfo.EncodesAsText {
		data = s.streamBuf.String()
	} else {
		data = s.streamBuf.Bytes()
	}
	if err := websocket.Message.Send(s.ws, data); err != nil {
		return fmt.Errorf("unable to send message %d: %s", data, err)
	}
	s.streamBuf.Reset()
	return nil
}

func createNonstreamEncoderDecoder(bc bulkConnection) (runtime.Encoder, runtime.Decoder) {
	si := bc.SerializerInfo()
	ns := bc.APIManager().negotiatedSerializer
	gv := bc.APIManager().GroupVersion
	decoder := ns.DecoderToVersion(si.Serializer, gv)
	encoder := ns.EncoderForVersion(si.Serializer, gv)
	return encoder, decoder
}

func handleWSClientMessage(ctx context.Context, bc bulkConnection, w http.ResponseWriter, req *http.Request) error {
	encoder, decoder := createNonstreamEncoderDecoder(bc)
	var err error
	handler := func(ws *websocket.Conn) {
		defer ws.Close()
		sink := &websocketSink{
			serializerInfo: bc.SerializerInfo(),
			encoder:        encoder,
			decoder:        decoder,
			bc:             bc,
			ws:             ws,
			requests:       make(chan *bulkapi.ClientMessage),
		}
		done := bc.StartProcessing(sink.requests, sink)
		err = sink.runReadAndProcessRequestsFromWSLoop(ctx, done)
	}

	websocket.Server{Handler: handler}.ServeHTTP(w, req)
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return err
	}
}
