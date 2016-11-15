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
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
)

type v2API interface {
	Post(ctx context.Context, r *pb.Request) (Response, error)
	Put(ctx context.Context, r *pb.Request) (Response, error)
	Delete(ctx context.Context, r *pb.Request) (Response, error)
	QGet(ctx context.Context, r *pb.Request) (Response, error)
	Get(ctx context.Context, r *pb.Request) (Response, error)
	Head(ctx context.Context, r *pb.Request) (Response, error)
}

type v2apiStore struct{ s *EtcdServer }

func (a *v2apiStore) Post(ctx context.Context, r *pb.Request) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *v2apiStore) Put(ctx context.Context, r *pb.Request) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *v2apiStore) Delete(ctx context.Context, r *pb.Request) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *v2apiStore) QGet(ctx context.Context, r *pb.Request) (Response, error) {
	return a.processRaftRequest(ctx, r)
}

func (a *v2apiStore) processRaftRequest(ctx context.Context, r *pb.Request) (Response, error) {
	data, err := r.Marshal()
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
		return resp, resp.err
	case <-ctx.Done():
		proposalsFailed.Inc()
		a.s.w.Trigger(r.ID, nil) // GC wait
		return Response{}, a.s.parseProposeCtxErr(ctx.Err(), start)
	case <-a.s.done:
	}
	return Response{}, ErrStopped
}

func (a *v2apiStore) Get(ctx context.Context, r *pb.Request) (Response, error) {
	if r.Wait {
		wc, err := a.s.store.Watch(r.Path, r.Recursive, r.Stream, r.Since)
		if err != nil {
			return Response{}, err
		}
		return Response{Watcher: wc}, nil
	}
	ev, err := a.s.store.Get(r.Path, r.Recursive, r.Sorted)
	if err != nil {
		return Response{}, err
	}
	return Response{Event: ev}, nil
}

func (a *v2apiStore) Head(ctx context.Context, r *pb.Request) (Response, error) {
	ev, err := a.s.store.Get(r.Path, r.Recursive, r.Sorted)
	if err != nil {
		return Response{}, err
	}
	return Response{Event: ev}, nil
}

// Do interprets r and performs an operation on s.store according to r.Method
// and other fields. If r.Method is "POST", "PUT", "DELETE", or a "GET" with
// Quorum == true, r will be sent through consensus before performing its
// respective operation. Do will block until an action is performed or there is
// an error.
func (s *EtcdServer) Do(ctx context.Context, r pb.Request) (Response, error) {
	r.ID = s.reqIDGen.Next()
	if r.Method == "GET" && r.Quorum {
		r.Method = "QGET"
	}
	v2api := (v2API)(&v2apiStore{s})
	switch r.Method {
	case "POST":
		return v2api.Post(ctx, &r)
	case "PUT":
		return v2api.Put(ctx, &r)
	case "DELETE":
		return v2api.Delete(ctx, &r)
	case "QGET":
		return v2api.QGet(ctx, &r)
	case "GET":
		return v2api.Get(ctx, &r)
	case "HEAD":
		return v2api.Head(ctx, &r)
	}
	return Response{}, ErrUnknownMethod
}
