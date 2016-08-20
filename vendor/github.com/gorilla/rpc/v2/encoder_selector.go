// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"io"
	"net/http"
)

// Encoder interface contains the encoder for http response.
// Eg. gzip, flate compressions.
type Encoder interface {
	Encode(w http.ResponseWriter) io.Writer
}

type encoder struct {
}

func (_ *encoder) Encode(w http.ResponseWriter) io.Writer {
	return w
}

var DefaultEncoder = &encoder{}

// EncoderSelector interface provides a way to select encoder using the http
// request. Typically people can use this to check HEADER of the request and
// figure out client capabilities.
// Eg. "Accept-Encoding" tells about supported compressions.
type EncoderSelector interface {
	Select(r *http.Request) Encoder
}

type encoderSelector struct {
}

func (_ *encoderSelector) Select(_ *http.Request) Encoder {
	return DefaultEncoder
}

var DefaultEncoderSelector = &encoderSelector{}
