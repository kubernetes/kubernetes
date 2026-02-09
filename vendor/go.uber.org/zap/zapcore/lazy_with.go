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
	core         Core
	originalCore Core
	sync.Once
	fields []Field
}

// NewLazyWith wraps a Core with a "lazy" Core that will only encode fields if
// the logger is written to (or is further chained in a lon-lazy manner).
func NewLazyWith(core Core, fields []Field) Core {
	return &lazyWithCore{
		core:         nil, // core is allocated once `initOnce` is called.
		originalCore: core,
		fields:       fields,
	}
}

func (d *lazyWithCore) initOnce() {
	d.Once.Do(func() {
		d.core = d.originalCore.With(d.fields)
	})
}

func (d *lazyWithCore) With(fields []Field) Core {
	d.initOnce()
	return d.core.With(fields)
}

func (d *lazyWithCore) Check(e Entry, ce *CheckedEntry) *CheckedEntry {
	// This is safe because `lazyWithCore` doesn't change the level.
	// So we can delagate the level check, any not `initOnce`
	// just for the check.
	if !d.originalCore.Enabled(e.Level) {
		return ce
	}
	d.initOnce()
	return d.core.Check(e, ce)
}

func (d *lazyWithCore) Enabled(level Level) bool {
	// Like above, this is safe because `lazyWithCore` doesn't change the level.
	return d.originalCore.Enabled(level)
}

func (d *lazyWithCore) Write(e Entry, fields []Field) error {
	d.initOnce()
	return d.core.Write(e, fields)
}

func (d *lazyWithCore) Sync() error {
	d.initOnce()
	return d.core.Sync()
}
