/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package interrupt

import (
	"os"
	"os/signal"
	"sync"
)

type Handler struct {
	notify []func()
	final  func(os.Signal)
	once   sync.Once
}

func Chain(handler *Handler, notify ...func()) *Handler {
	if handler == nil {
		return New(nil, notify...)
	}
	return New(handler.Signal, append(notify, handler.Close)...)
}

func New(final func(os.Signal), notify ...func()) *Handler {
	return &Handler{
		final:  final,
		notify: notify,
	}
}

func (h *Handler) Close() {
	h.once.Do(func() {
		for _, fn := range h.notify {
			fn()
		}
	})
}

func (h *Handler) Signal(s os.Signal) {
	h.once.Do(func() {
		for _, fn := range h.notify {
			fn()
		}
		if h.final == nil {
			os.Exit(0)
		}
		h.final(s)
	})
}

func (h *Handler) Run(fn func() error) error {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, childSignals...)
	defer signal.Stop(ch)
	go func() {
		sig, ok := <-ch
		if !ok {
			return
		}
		h.Signal(sig)
	}()
	defer h.Close()
	return fn()
}
