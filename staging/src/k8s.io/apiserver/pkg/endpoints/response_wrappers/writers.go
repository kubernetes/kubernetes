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

package response_wrappers

import (
	"bufio"
	"net"
	"net/http"
)

// Wrap wraps the given middle ResponseWriter to make it also
// implement http.CloseNotifier and http.Flusher and http.Hijacker if
// the given inner ResponseWriter implements all three, implement
// http.CloseNotifier and http.Flusher if inner implements those two,
// otherwise add nothing.
// The behavior added to middle is centered on delegating to inner
// and may be augmented by the optional functional parameters here.
// - beforeCloseNotify, if not nil, is called before inner.CloseNotify and
//   if it returns a non-nil value that is returned instead of calling on inner.
// - afterCloseNofity, if not nil, is given the channel returned by inner and
//   returns the channel for the wrapper to return.
// - beforeFlush, if not nil, is called before inner.Flush.
// - afterFlush, if not nil, is called after inner.Flush.
// - beforeHijack, if not nil, is called before inner.Hijack and if it
//   returns an error then that is returned with nils instead of invoking inner.Hijack.
// - afterHijack, if not nil, is called after inner.Hijack with the returned values from
//   that and returns the values for the wrapper to return.
func Wrap(inner http.ResponseWriter,
	middle http.ResponseWriter,
	beforeCloseNotify func() <-chan bool,
	afterCloseNofity func(<-chan bool) <-chan bool,
	beforeFlush func(),
	afterFlush func(),
	beforeHijack func() error,
	afterHijack func(net.Conn, *bufio.ReadWriter, error) (net.Conn, *bufio.ReadWriter, error),
) http.ResponseWriter {
	//lint:ignore SA1019 above this line's pay grade
	if _, cn := inner.(http.CloseNotifier); !cn {
		return middle
	}
	if _, fl := inner.(http.Flusher); !fl {
		return middle
	}
	wrap1 := withCloseNotifyAndFlush{
		ResponseWriter:    middle,
		inner:             inner.(ResponseWriterAndCloseNotifierAndFlusher),
		beforeCloseNotify: beforeCloseNotify,
		afterCloseNofity:  afterCloseNofity,
		beforeFlush:       beforeFlush,
		afterFlush:        afterFlush,
	}
	if _, hj := inner.(http.Hijacker); hj {
		return &withCloseNotifyAndFlushAndHijack{
			withCloseNotifyAndFlush: wrap1,
			beforeHijack:            beforeHijack,
			afterHijack:             afterHijack,
		}
	}
	return &wrap1
}

type withCloseNotifyAndFlush struct {
	http.ResponseWriter
	inner             ResponseWriterAndCloseNotifierAndFlusher
	beforeCloseNotify func() <-chan bool
	afterCloseNofity  func(<-chan bool) <-chan bool
	beforeFlush       func()
	afterFlush        func()
}

type ResponseWriterAndCloseNotifierAndFlusher interface {
	http.ResponseWriter
	//lint:ignore SA1019 above this line's pay grade
	http.CloseNotifier
	http.Flusher
}

//lint:ignore SA1019 above this line's pay grade
var _ http.CloseNotifier = &withCloseNotifyAndFlush{}
var _ http.Flusher = &withCloseNotifyAndFlush{}

func (wr *withCloseNotifyAndFlush) CloseNotify() <-chan bool {
	if wr.beforeCloseNotify != nil {
		ans := wr.beforeCloseNotify()
		if ans != nil {
			return ans
		}
	}
	ans := wr.inner.CloseNotify()
	if wr.afterCloseNofity != nil {
		return wr.afterCloseNofity(ans)
	}
	return ans
}

func (wr *withCloseNotifyAndFlush) Flush() {
	if wr.beforeFlush != nil {
		wr.beforeFlush()
	}
	wr.inner.Flush()
	if wr.afterFlush != nil {
		wr.afterFlush()
	}
}

type withCloseNotifyAndFlushAndHijack struct {
	withCloseNotifyAndFlush
	beforeHijack func() error
	afterHijack  func(net.Conn, *bufio.ReadWriter, error) (net.Conn, *bufio.ReadWriter, error)
}

//lint:ignore SA1019 above this line's pay grade
var _ http.CloseNotifier = &withCloseNotifyAndFlushAndHijack{}
var _ http.Flusher = &withCloseNotifyAndFlushAndHijack{}
var _ http.Hijacker = &withCloseNotifyAndFlushAndHijack{}

func (wr *withCloseNotifyAndFlushAndHijack) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if wr.beforeHijack != nil {
		err := wr.beforeHijack()
		if err != nil {
			return nil, nil, err
		}
	}
	conn, rw, err := wr.inner.(http.Hijacker).Hijack()
	if wr.afterHijack != nil {
		return wr.afterHijack(conn, rw, err)
	}
	return conn, rw, err
}
