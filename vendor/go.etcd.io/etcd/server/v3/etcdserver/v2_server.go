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

package etcdserver

import (
	"context"
	"time"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
)

type RequestV2 pb.Request

type RequestV2Handler interface {
	Post(ctx context.Context, r *RequestV2) (Response, error)
	Put(ctx context.Context, r *RequestV2) (Response, error)
	Delete(ctx context.Context, r *RequestV2) (Response, error)
	QGet(ctx context.Context, r *RequestV2) (Response, error)
	Get(ctx context.Context, r *RequestV2) (Response, error)
	Head(ctx context.Context, r *RequestV2) (Response, error)
}

type reqV2HandlerEtcdServer struct {
	reqV2HandlerStore
	s *EtcdServer
}

type reqV2HandlerStore struct {
	store   v2store.Store
	applier ApplierV2
}

func NewStoreRequestV2Handler(s v2store.Store, applier ApplierV2) RequestV2Handler {
	return &reqV2HandlerStore{s, applier}
}

func (a *reqV2HandlerStore) Post(ctx context.Context, r *RequestV2) (Response, error) {
	return a.applier.Post(r), nil
}

func (a *reqV2HandlerStore) Put(ctx context.Context, r *RequestV2) (Response, error) {
	return a.applier.Put(r), nil
}

func (a *reqV2HandlerStore) Delete(ctx context.Context, r *RequestV2) (Response, error) {
	return a.applier.Delete(r), nil
}

func (a *reqV2HandlerStore) QGet(ctx context.Context, r *RequestV2) (Response, error) {
	return a.applier.QGet(r), nil
}

func (a *reqV2HandlerStore) Get(ctx context.Context, r *RequestV2) (Response, error) {
	if r.Wait {
		wc, err := a.store.Watch(r.Path, r.Recursive, r.Stream, r.Since)
		return Response{Watcher: wc}, err
	}
	ev, err := a.store.Get(r.Path, r.Recursive, r.Sorted)
	return Response{Event: ev}, err
}

func (a *reqV2HandlerStore) Head(ctx context.Context, r *RequestV2) (Response, error) {
	ev, err := a.store.Get(r.Path, r.Recursive, r.Sorted)
	return Response{Event: ev}, err
}

func (a *reqV2HandlerEtcdServer) Post(ctx context.Context, r *RequestV2) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *reqV2HandlerEtcdServer) Put(ctx context.Context, r *RequestV2) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *reqV2HandlerEtcdServer) Delete(ctx context.Context, r *RequestV2) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *reqV2HandlerEtcdServer) QGet(ctx context.Context, r *RequestV2) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *reqV2HandlerEtcdServer) processRaftRequest(ctx context.Context, r *RequestV2) (Response, error) {
	data, err := ((*pb.Request)(r)).Marshal()
	if err != nil {
		return Response{}, err
	}
	ch := a.s.w.Register(r.ID)

	start := time.Now()
	a.s.r.Propose(ctx, data)
	proposalsPending.Inc()
	defer proposalsPending.Dec()

	select {
	case x := <-ch:
		resp := x.(Response)
		return resp, resp.Err
	case <-ctx.Done():
		proposalsFailed.Inc()
		a.s.w.Trigger(r.ID, nil) // GC wait
		return Response{}, a.s.parseProposeCtxErr(ctx.Err(), start)
	case <-a.s.stopping:
	}
	return Response{}, ErrStopped
}

func (s *EtcdServer) Do(ctx context.Context, r pb.Request) (Response, error) {
	r.ID = s.reqIDGen.Next()
	h := &reqV2HandlerEtcdServer{
		reqV2HandlerStore: reqV2HandlerStore{
			store:   s.v2store,
			applier: s.applyV2,
		},
		s: s,
	}
	rp := &r
	resp, err := ((*RequestV2)(rp)).Handle(ctx, h)
	resp.Term, resp.Index = s.Term(), s.CommittedIndex()
	return resp, err
}

// Handle interprets r and performs an operation on s.store according to r.Method
// and other fields. If r.Method is "POST", "PUT", "DELETE", or a "GET" with
// Quorum == true, r will be sent through consensus before performing its
// respective operation. Do will block until an action is performed or there is
// an error.
func (r *RequestV2) Handle(ctx context.Context, v2api RequestV2Handler) (Response, error) {
	if r.Method == "GET" && r.Quorum {
		r.Method = "QGET"
	}
	switch r.Method {
	case "POST":
		return v2api.Post(ctx, r)
	case "PUT":
		return v2api.Put(ctx, r)
	case "DELETE":
		return v2api.Delete(ctx, r)
	case "QGET":
		return v2api.QGet(ctx, r)
	case "GET":
		return v2api.Get(ctx, r)
	case "HEAD":
		return v2api.Head(ctx, r)
	}
	return Response{}, ErrUnknownMethod
}

func (r *RequestV2) String() string {
	rpb := pb.Request(*r)
	return rpb.String()
}
