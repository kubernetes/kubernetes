// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json2

import (
	"errors"
)

type ErrorCode int

const (
	E_PARSE       ErrorCode = -32700
	E_INVALID_REQ ErrorCode = -32600
	E_NO_METHOD   ErrorCode = -32601
	E_BAD_PARAMS  ErrorCode = -32602
	E_INTERNAL    ErrorCode = -32603
	E_SERVER      ErrorCode = -32000
)

var ErrNullResult = errors.New("result is null")

type Error struct {
	// A Number that indicates the error type that occurred.
	Code ErrorCode `json:"code"` /* required */

	// A String providing a short description of the error.
	// The message SHOULD be limited to a concise single sentence.
	Message string `json:"message"` /* required */

	// A Primitive or Structured value that contains additional information about the error.
	Data interface{} `json:"data"` /* optional */
}

func (e *Error) Error() string {
	return e.Message
}
