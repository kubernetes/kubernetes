/*
Copyright 2021 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"sync"
)

type priorityAndFairnessKeyType int

const (
	// priorityAndFairnessInitializationSignalKey is a key under which
	// initialization signal function for watch requests is stored
	// in the context.
	priorityAndFairnessInitializationSignalKey priorityAndFairnessKeyType = iota
)

// WithInitializationSignal creates a copy of parent context with
// priority and fairness initialization signal value.
func WithInitializationSignal(ctx context.Context, signal InitializationSignal) context.Context {
	return context.WithValue(ctx, priorityAndFairnessInitializationSignalKey, signal)
}

// initializationSignalFrom returns an initialization signal function
// which when called signals that watch initialization has already finished
// to priority and fairness dispatcher.
func initializationSignalFrom(ctx context.Context) (InitializationSignal, bool) {
	signal, ok := ctx.Value(priorityAndFairnessInitializationSignalKey).(InitializationSignal)
	return signal, ok && signal != nil
}

// WatchInitialized sends a signal to priority and fairness dispatcher
// that a given watch request has already been initialized.
func WatchInitialized(ctx context.Context) {
	if signal, ok := initializationSignalFrom(ctx); ok {
		signal.Signal()
	}
}

// RequestDelegated informs the priority and fairness dispatcher that
// a given request has been delegated to an aggregated API
// server. No-op when priority and fairness is disabled.
func RequestDelegated(ctx context.Context) {
	// The watch initialization signal doesn't traverse request
	// boundaries, so we generously fire it as soon as we know
	// that the request won't be serviced locally. Safe to call
	// for non-watch requests.
	WatchInitialized(ctx)
}

// InitializationSignal is an interface that allows sending and handling
// initialization signals.
type InitializationSignal interface {
	// Signal notifies the dispatcher about finished initialization.
	Signal()
	// Wait waits for the initialization signal.
	Wait()
}

type initializationSignal struct {
	once sync.Once
	done chan struct{}
}

func NewInitializationSignal() InitializationSignal {
	return &initializationSignal{
		once: sync.Once{},
		done: make(chan struct{}),
	}
}

func (i *initializationSignal) Signal() {
	i.once.Do(func() { close(i.done) })
}

func (i *initializationSignal) Wait() {
	<-i.done
}
