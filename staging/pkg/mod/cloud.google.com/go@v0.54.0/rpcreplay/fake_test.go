// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rpcreplay

import (
	"context"
	"io"
	"log"
	"net"

	pb "cloud.google.com/go/rpcreplay/proto/intstore"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// intStoreServer is an in-memory implementation of IntStore.
type intStoreServer struct {
	pb.IntStoreServer

	Addr string
	l    net.Listener
	gsrv *grpc.Server

	items map[string]int32
}

func newIntStoreServer() *intStoreServer {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	s := &intStoreServer{
		Addr: l.Addr().String(),
		l:    l,
		gsrv: grpc.NewServer(),
	}
	pb.RegisterIntStoreServer(s.gsrv, s)
	go s.gsrv.Serve(s.l)
	return s
}

func (s *intStoreServer) stop() {
	s.gsrv.Stop()
	s.l.Close()
}

func (s *intStoreServer) Set(_ context.Context, item *pb.Item) (*pb.SetResponse, error) {
	old := s.setItem(item)
	return &pb.SetResponse{PrevValue: old}, nil
}

func (s *intStoreServer) setItem(item *pb.Item) int32 {
	if s.items == nil {
		s.items = map[string]int32{}
	}
	old := s.items[item.Name]
	s.items[item.Name] = item.Value
	return old
}

func (s *intStoreServer) Get(_ context.Context, req *pb.GetRequest) (*pb.Item, error) {
	val, ok := s.items[req.Name]
	if !ok {
		return nil, status.Errorf(codes.NotFound, "%q", req.Name)
	}
	return &pb.Item{Name: req.Name, Value: val}, nil
}

func (s *intStoreServer) ListItems(req *pb.ListItemsRequest, ss pb.IntStore_ListItemsServer) error {
	for name, val := range s.items {
		if val > req.GreaterThan {
			if err := ss.Send(&pb.Item{Name: name, Value: val}); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *intStoreServer) SetStream(ss pb.IntStore_SetStreamServer) error {
	n := 0
	for {
		item, err := ss.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		s.setItem(item)
		n++
	}
	return ss.SendAndClose(&pb.Summary{Count: int32(n)})
}

func (s *intStoreServer) StreamChat(ss pb.IntStore_StreamChatServer) error {
	for {
		item, err := ss.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if err := ss.Send(item); err != nil {
			return err
		}
	}
	return nil
}
