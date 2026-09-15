/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"context"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// WatchRecorder collects events from a background watch stream.
type WatchRecorder struct {
	w      watch.Interface
	req    WatchRequest
	events []watch.Event
	mu     sync.Mutex
	done   chan struct{}
}

// NewWatchRecorder starts a background watcher with the specified WatchRequest.
func NewWatchRecorder(ctx context.Context, store storage.Interface, req WatchRequest) (*WatchRecorder, error) {
	req.Predicate = normalizePredicate(req.Predicate)
	recursive := req.Recursive || strings.HasSuffix(req.Key, "/")
	opts := storage.ListOptions{
		ResourceVersion: req.ResourceVersion,
		Predicate:       req.Predicate,
		Recursive:       recursive,
		ProgressNotify:  true,
	}
	sendInitial := req.SendInitialEvents
	opts.SendInitialEvents = &sendInitial

	w, err := store.Watch(ctx, req.Key, opts)
	if err != nil {
		return nil, err
	}
	rec := &WatchRecorder{
		w:    w,
		req:  req,
		done: make(chan struct{}),
	}
	go rec.collect()
	return rec, nil
}

func (r *WatchRecorder) collect() {
	defer close(r.done)
	for ev := range r.w.ResultChan() {
		r.mu.Lock()
		r.events = append(r.events, ev)
		r.mu.Unlock()
	}
}

// Request returns the watch request configuration.
func (r *WatchRecorder) Request() WatchRequest {
	return r.req
}

// Response returns the recorded watch response containing all collected events so far.
func (r *WatchRecorder) Response() WatchResponse {
	return WatchResponse{
		Events: r.Events(),
	}
}

// Events returns a snapshot copy of all events collected so far.
func (r *WatchRecorder) Events() []watch.Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	copied := make([]watch.Event, len(r.events))
	copy(copied, r.events)
	return copied
}

// RecordedWatch returns a RecordedWatch snapshot for verification.
func (r *WatchRecorder) RecordedWatch() RecordedWatch {
	return RecordedWatch{
		Request:  r.Request(),
		Response: r.Response(),
	}
}

// Stop closes the watch stream and waits for event ingestion to finish.
func (r *WatchRecorder) Stop() {
	r.w.Stop()
	<-r.done
}
