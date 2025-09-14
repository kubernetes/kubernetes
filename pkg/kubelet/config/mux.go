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

package config

import (
	"context"
	"sync"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

type merger interface {
	// Invoked when a change from a source is received.  May also function as an incremental
	// merger if you wish to consume changes incrementally.  Must be reentrant when more than
	// one source is defined.
	Merge(ctx context.Context, source string, update interface{}) error
}

// mux is a class for merging configuration from multiple sources.  Changes are
// pushed via channels and sent to the merge function.
type mux struct {
	// Invoked when an update is sent to a source.
	merger merger

	// Sources and their lock.
	sourceLock sync.RWMutex
	// Maps source names to channels
	sources map[string]chan interface{}
}

// newMux creates a new mux that can merge changes from multiple sources.
func newMux(merger merger) *mux {
	mux := &mux{
		sources: make(map[string]chan interface{}),
		merger:  merger,
	}
	return mux
}

// ChannelWithContext returns a channel where a configuration source
// can send updates of new configurations. Multiple calls with the same
// source will return the same channel. This allows change and state based sources
// to use the same channel. Different source names however will be treated as a
// union.
func (m *mux) ChannelWithContext(ctx context.Context, source string) chan interface{} {
	if len(source) == 0 {
		panic("Channel given an empty name")
	}
	m.sourceLock.Lock()
	defer m.sourceLock.Unlock()
	channel, exists := m.sources[source]
	if exists {
		return channel
	}
	newChannel := make(chan interface{})
	m.sources[source] = newChannel

	go wait.Until(func() { m.listen(ctx, source, newChannel) }, 0, ctx.Done())
	return newChannel
}

func (m *mux) listen(ctx context.Context, source string, listenChannel <-chan interface{}) {
	logger := klog.FromContext(ctx)
	for update := range listenChannel {
		if err := m.merger.Merge(ctx, source, update); err != nil {
			logger.Info("failed merging update", "err", err)
		}
	}
}
