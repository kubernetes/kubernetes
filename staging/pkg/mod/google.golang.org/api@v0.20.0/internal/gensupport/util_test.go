// Copyright 2016 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"io"
	"time"
)

// errReader reads out of a buffer until it is empty, then returns the specified error.
type errReader struct {
	buf []byte
	err error
}

func (er *errReader) Read(p []byte) (int, error) {
	if len(er.buf) == 0 {
		if er.err == nil {
			return 0, io.EOF
		}
		return 0, er.err
	}
	n := copy(p, er.buf)
	er.buf = er.buf[n:]
	return n, nil
}

// NoPauseBackoff implements backoff with infinite 0-length pauses.
type NoPauseBackoff struct{}

func (bo *NoPauseBackoff) Pause() time.Duration { return 0 }

// PauseOneSecond implements backoff with infinite 1s pauses.
type PauseOneSecond struct{}

func (bo *PauseOneSecond) Pause() time.Duration { return time.Second }

// PauseForeverBackoff implements backoff with infinite 1h pauses.
type PauseForeverBackoff struct{}

func (bo *PauseForeverBackoff) Pause() time.Duration { return time.Hour }
