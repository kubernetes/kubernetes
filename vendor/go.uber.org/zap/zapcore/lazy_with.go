// Copyright (c) 2023 Uber Technologies, Inc.
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

package zapcore

import "sync"

type lazyWithCore struct {
	Core
	sync.Once
	fields []Field
}

// NewLazyWith wraps a Core with a "lazy" Core that will only encode fields if
// the logger is written to (or is further chained in a lon-lazy manner).
func NewLazyWith(core Core, fields []Field) Core {
	return &lazyWithCore{
		Core:   core,
		fields: fields,
	}
}

func (d *lazyWithCore) initOnce() {
	d.Once.Do(func() {
		d.Core = d.Core.With(d.fields)
	})
}

func (d *lazyWithCore) With(fields []Field) Core {
	d.initOnce()
	return d.Core.With(fields)
}

func (d *lazyWithCore) Check(e Entry, ce *CheckedEntry) *CheckedEntry {
	d.initOnce()
	return d.Core.Check(e, ce)
}
