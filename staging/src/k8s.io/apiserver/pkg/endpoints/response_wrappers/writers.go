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

// Wrap037 wraps the given middle ResponseWriter to make it also
// implement some extension interfaces depending on which
// of those are also implemented by the given inner ResponseWriter.
// Of the 8 possible cases, this func recognizes the following three.
// -- Implement http.CloseNotifier, http.Flusher, and http.Hijacker if inner does.
// -- Implement http.CloseNotifier and http.Flusher if inner does.
// -- Otherwise add nothing.
// The behavior added to middle is centered on delegating to inner
// and may be augmented by the optional functional parameters here.
// - closeNotify, if not nil, is used in place of inner.CloseNotify
//   and is given that func as a parameter.
// - flush, if not nil, is used in place of inner.Flush
//   and is given that func as a parameter.
// - hijack, if not nil, is used in place of inner.Hijack
//   and is given that func as a parameter.
func Wrap037(inner http.ResponseWriter,
	middle http.ResponseWriter,
	closeNotify func(func() <-chan bool) <-chan bool,
	flush func(func()),
	hijack func(func() (net.Conn, *bufio.ReadWriter, error)) (net.Conn, *bufio.ReadWriter, error),
) http.ResponseWriter {
	wrap03 := Wrap03(inner, middle, closeNotify, flush)
	return Wrap03To037(inner, wrap03, hijack)
}

// Wrap03 wraps the given middle ResponseWriter to make it also
// implement some extension interfaces depending on which
// of those are also implemented by the given inner ResponseWriter.
// Of the 4 possible cases, this func recognizes the following two.
// -- Implement http.CloseNotifier and http.Flusher if inner does.
// -- Otherwise add nothing.
// The behavior added to middle is centered on delegating to inner
// and may be augmented by the optional functional parameters here.
// - closeNotify, if not nil, is used in place of inner.CloseNotify
//   and is given that func as a parameter.
// - flush, if not nil, is used in place of inner.Flush
//   and is given that func as a parameter.
func Wrap03(inner http.ResponseWriter,
	middle http.ResponseWriter,
	closeNotify func(func() <-chan bool) <-chan bool,
	flush func(func()),
) http.ResponseWriter {
	narrowed, ok := inner.(closeNotifierAndFlusher)
	if !ok {
		return middle
	}
	return &withCloseNotifyAndFlush{
		ResponseWriter: middle,
		inner:          narrowed,
		closeNotify:    closeNotify,
		flush:          flush,
	}
}

type withCloseNotifyAndFlush struct {
	http.ResponseWriter
	inner       closeNotifierAndFlusher
	closeNotify func(func() <-chan bool) <-chan bool
	flush       func(func())
}

type closeNotifierAndFlusher interface {
	//lint:ignore SA1019 above this line's pay grade
	http.CloseNotifier
	http.Flusher
}

//lint:ignore SA1019 above this line's pay grade
var _ http.CloseNotifier = &withCloseNotifyAndFlush{}
var _ http.Flusher = &withCloseNotifyAndFlush{}

func (wr *withCloseNotifyAndFlush) CloseNotify() <-chan bool {
	if wr.closeNotify != nil {
		return wr.closeNotify(wr.inner.CloseNotify)
	}
	return wr.inner.CloseNotify()
}

func (wr *withCloseNotifyAndFlush) Flush() {
	if wr.flush != nil {
		wr.flush(wr.inner.Flush)
	} else {
		wr.inner.Flush()

	}
}

// Wrap03To037 wraps the given middle ResponseWriter to make it also
// implement some extension interfaces depending on which
// of those are also implemented by the given middle and inner ResponseWriters.
// Of the possible cases, this func recognizes the following two.
// -- Add http.Hijacker if inner implements it and middle implements
//    http.CloseNotifier and http.Flusher.
// -- Otherwise add nothing.
// The behavior added to middle is centered on delegating to inner
// and may be augmented by the optional functional parameters here.
// - hijack, if not nil, is used in place of inner.Hijack
//   and is given that func as a parameter.
func Wrap03To037(inner http.ResponseWriter,
	middle http.ResponseWriter,
	hijack func(func() (net.Conn, *bufio.ReadWriter, error)) (net.Conn, *bufio.ReadWriter, error),
) http.ResponseWriter {
	middleNarrowed, ok := middle.(responseWriterAndCloseNotifierAndFlusher)
	if !ok {
		return middle
	}
	if innerNarrowed, hj := inner.(http.Hijacker); hj {
		return &closeNotifierAndFlusherWithHijack{
			responseWriterAndCloseNotifierAndFlusher: middleNarrowed,
			inner:                                    innerNarrowed,
			hijack:                                   hijack,
		}
	}
	return middle
}

type closeNotifierAndFlusherWithHijack struct {
	responseWriterAndCloseNotifierAndFlusher
	inner  http.Hijacker
	hijack func(func() (net.Conn, *bufio.ReadWriter, error)) (net.Conn, *bufio.ReadWriter, error)
}

type responseWriterAndCloseNotifierAndFlusher interface {
	http.ResponseWriter
	closeNotifierAndFlusher
}

//lint:ignore SA1019 above this line's pay grade
var _ http.CloseNotifier = &closeNotifierAndFlusherWithHijack{}
var _ http.Flusher = &closeNotifierAndFlusherWithHijack{}
var _ http.Hijacker = &closeNotifierAndFlusherWithHijack{}

func (wr *closeNotifierAndFlusherWithHijack) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if wr.hijack != nil {
		return wr.hijack(wr.inner.Hijack)
	}
	return wr.inner.Hijack()
}
