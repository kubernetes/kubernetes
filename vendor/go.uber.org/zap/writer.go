// Copyright (c) 2016-2022 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zap

import (
	"fmt"
	"io"

	"go.uber.org/zap/zapcore"

	"go.uber.org/multierr"
)

// Open is a high-level wrapper that takes a variadic number of URLs, opens or
// creates each of the specified resources, and combines them into a locked
// WriteSyncer. It also returns any error encountered and a function to close
// any opened files.
//
// Passing no URLs returns a no-op WriteSyncer. Zap handles URLs without a
// scheme and URLs with the "file" scheme. Third-party code may register
// factories for other schemes using RegisterSink.
//
// URLs with the "file" scheme must use absolute paths on the local
// filesystem. No user, password, port, fragments, or query parameters are
// allowed, and the hostname must be empty or "localhost".
//
// Since it's common to write logs to the local filesystem, URLs without a
// scheme (e.g., "/var/log/foo.log") are treated as local file paths. Without
// a scheme, the special paths "stdout" and "stderr" are interpreted as
// os.Stdout and os.Stderr. When specified without a scheme, relative file
// paths also work.
func Open(paths ...string) (zapcore.WriteSyncer, func(), error) {
	writers, closeAll, err := open(paths)
	if err != nil {
		return nil, nil, err
	}

	writer := CombineWriteSyncers(writers...)
	return writer, closeAll, nil
}

func open(paths []string) ([]zapcore.WriteSyncer, func(), error) {
	writers := make([]zapcore.WriteSyncer, 0, len(paths))
	closers := make([]io.Closer, 0, len(paths))
	closeAll := func() {
		for _, c := range closers {
			_ = c.Close()
		}
	}

	var openErr error
	for _, path := range paths {
		sink, err := _sinkRegistry.newSink(path)
		if err != nil {
			openErr = multierr.Append(openErr, fmt.Errorf("open sink %q: %w", path, err))
			continue
		}
		writers = append(writers, sink)
		closers = append(closers, sink)
	}
	if openErr != nil {
		closeAll()
		return nil, nil, openErr
	}

	return writers, closeAll, nil
}

// CombineWriteSyncers is a utility that combines multiple WriteSyncers into a
// single, locked WriteSyncer. If no inputs are supplied, it returns a no-op
// WriteSyncer.
//
// It's provided purely as a convenience; the result is no different from
// using zapcore.NewMultiWriteSyncer and zapcore.Lock individually.
func CombineWriteSyncers(writers ...zapcore.WriteSyncer) zapcore.WriteSyncer {
	if len(writers) == 0 {
		return zapcore.AddSync(io.Discard)
	}
	return zapcore.Lock(zapcore.NewMultiWriteSyncer(writers...))
}
