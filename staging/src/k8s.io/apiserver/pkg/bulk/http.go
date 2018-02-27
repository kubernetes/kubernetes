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
	"fmt"
	"io"
	"net/http"

	"golang.org/x/net/context"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
)

type httpSink struct {
	flusher        http.Flusher
	streamEncoder  streaming.Encoder
	serializerInfo runtime.SerializerInfo
	requests       chan *bulkapi.BulkRequest
}

func (s *httpSink) WriteResponse(resp *bulkapi.BulkResponse) error {
	if err := s.streamEncoder.Encode(resp); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

func handleHTTPBulkRequest(ctx context.Context, bc bulkConnection, w http.ResponseWriter, req *http.Request) error {
	si := bc.SerializerInfo()
	if si.StreamSerializer == nil || si.StreamSerializer.Framer == nil {
		return fmt.Errorf("no stream framing support is available for media type %q", si.MediaType)
	}
	frameWriter := si.StreamSerializer.Framer.NewFrameWriter(w)
	frameReader := si.StreamSerializer.Framer.NewFrameReader(req.Body)

	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("unable to process bulk request - can't get http.Flusher: %#v", w)
	}
	cn, ok := w.(http.CloseNotifier)
	if !ok {
		return fmt.Errorf("unable to process bulk request - can't get http.CloseNotifier: %#v", w)
	}

	decoder := bc.APIManager().negotiatedSerializer.DecoderToVersion(si.StreamSerializer, bc.APIManager().GroupVersion)
	encoder := bc.APIManager().negotiatedSerializer.EncoderForVersion(si.StreamSerializer, bc.APIManager().GroupVersion)
	streamDecoder := streaming.NewDecoder(frameReader, decoder)
	streamEncoder := streaming.NewEncoder(frameWriter, encoder)

	requests, err := readBulkReqeusts(bc, streamDecoder)
	if err != nil {
		return err
	}
	sink := &httpSink{
		streamEncoder:  streamEncoder,
		serializerInfo: si,
		flusher:        flusher,
		requests:       make(chan *bulkapi.BulkRequest),
	}

	w.Header().Set("Transfer-Encoding", "chunked")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	// no errors may be returned after sending OK

	done := bc.StartProcessing(sink.requests, sink)

	for _, req := range requests {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case sink.requests <- req:
		}
	}
	close(sink.requests)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		// all requests are processed
		return nil
	case <-cn.CloseNotify():
		// client closed connection
		return nil
	}
}

func readBulkReqeusts(bc bulkConnection, sd streaming.Decoder) (result []*bulkapi.BulkRequest, err error) {
	defer sd.Close()
	defaultGVK := bc.APIManager().GroupVersion.WithKind("BulkRequest")
	for {
		var obj runtime.Object = &bulkapi.BulkRequest{}
		obj, _, err = sd.Decode(&defaultGVK, obj)
		if err == io.EOF {
			return result, nil
		}
		if err != nil {
			return
		}
		br, ok := obj.(*bulkapi.BulkRequest)
		if !ok {
			err = fmt.Errorf("unable to decode bulk request: cast error")
			return
		}
		result = append(result, br)
	}
}
