// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package grpcproxy

import (
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

type ws2wc struct{ wserv pb.WatchServer }

func WatchServerToWatchClient(wserv pb.WatchServer) pb.WatchClient {
	return &ws2wc{wserv}
}

func (s *ws2wc) Watch(ctx context.Context, opts ...grpc.CallOption) (pb.Watch_WatchClient, error) {
	ch1, ch2 := make(chan interface{}), make(chan interface{})
	headerc, trailerc := make(chan metadata.MD, 1), make(chan metadata.MD, 1)
	wclient := &ws2wcClientStream{chanClientStream{headerc, trailerc, &chanStream{ch1, ch2, ctx}}}
	wserver := &ws2wcServerStream{chanServerStream{headerc, trailerc, &chanStream{ch2, ch1, ctx}}}
	go func() {
		s.wserv.Watch(wserver)
		// close the server side sender
		close(ch1)
	}()
	return wclient, nil
}

// ws2wcClientStream implements Watch_WatchClient
type ws2wcClientStream struct{ chanClientStream }

// ws2wcServerStream implements Watch_WatchServer
type ws2wcServerStream struct{ chanServerStream }

func (s *ws2wcClientStream) Send(wr *pb.WatchRequest) error {
	return s.SendMsg(wr)
}
func (s *ws2wcClientStream) Recv() (*pb.WatchResponse, error) {
	var v interface{}
	if err := s.RecvMsg(&v); err != nil {
		return nil, err
	}
	return v.(*pb.WatchResponse), nil
}

func (s *ws2wcServerStream) Send(wr *pb.WatchResponse) error {
	return s.SendMsg(wr)
}
func (s *ws2wcServerStream) Recv() (*pb.WatchRequest, error) {
	var v interface{}
	if err := s.RecvMsg(&v); err != nil {
		return nil, err
	}
	return v.(*pb.WatchRequest), nil
}

// chanServerStream implements grpc.ServerStream with a chanStream
type chanServerStream struct {
	headerc  chan<- metadata.MD
	trailerc chan<- metadata.MD
	grpc.Stream
}

func (ss *chanServerStream) SendHeader(md metadata.MD) error {
	select {
	case ss.headerc <- md:
		return nil
	case <-ss.Context().Done():
	}
	return ss.Context().Err()
}

func (ss *chanServerStream) SetTrailer(md metadata.MD) {
	ss.trailerc <- md
}

// chanClientStream implements grpc.ClientStream with a chanStream
type chanClientStream struct {
	headerc  <-chan metadata.MD
	trailerc <-chan metadata.MD
	*chanStream
}

func (cs *chanClientStream) Header() (metadata.MD, error) {
	select {
	case md := <-cs.headerc:
		return md, nil
	case <-cs.Context().Done():
	}
	return nil, cs.Context().Err()
}

func (cs *chanClientStream) Trailer() metadata.MD {
	select {
	case md := <-cs.trailerc:
		return md
	case <-cs.Context().Done():
		return nil
	}
}

func (s *chanClientStream) CloseSend() error {
	close(s.chanStream.sendc)
	return nil
}

// chanStream implements grpc.Stream using channels
type chanStream struct {
	recvc <-chan interface{}
	sendc chan<- interface{}
	ctx   context.Context
}

func (s *chanStream) Context() context.Context { return s.ctx }

func (s *chanStream) SendMsg(m interface{}) error {
	select {
	case s.sendc <- m:
		return nil
	case <-s.ctx.Done():
	}
	return s.ctx.Err()
}

func (s *chanStream) RecvMsg(m interface{}) error {
	v := m.(*interface{})
	select {
	case msg, ok := <-s.recvc:
		if !ok {
			return grpc.ErrClientConnClosing
		}
		*v = msg
		return nil
	case <-s.ctx.Done():
	}
	return s.ctx.Err()
}
