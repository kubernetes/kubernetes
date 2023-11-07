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

package zapcore

import "go.uber.org/multierr"

type multiCore []Core

var (
	_ leveledEnabler = multiCore(nil)
	_ Core           = multiCore(nil)
)

// NewTee creates a Core that duplicates log entries into two or more
// underlying Cores.
//
// Calling it with a single Core returns the input unchanged, and calling
// it with no input returns a no-op Core.
func NewTee(cores ...Core) Core {
	switch len(cores) {
	case 0:
		return NewNopCore()
	case 1:
		return cores[0]
	default:
		return multiCore(cores)
	}
}

func (mc multiCore) With(fields []Field) Core {
	clone := make(multiCore, len(mc))
	for i := range mc {
		clone[i] = mc[i].With(fields)
	}
	return clone
}

func (mc multiCore) Level() Level {
	minLvl := _maxLevel // mc is never empty
	for i := range mc {
		if lvl := LevelOf(mc[i]); lvl < minLvl {
			minLvl = lvl
		}
	}
	return minLvl
}

func (mc multiCore) Enabled(lvl Level) bool {
	for i := range mc {
		if mc[i].Enabled(lvl) {
			return true
		}
	}
	return false
}

func (mc multiCore) Check(ent Entry, ce *CheckedEntry) *CheckedEntry {
	for i := range mc {
		ce = mc[i].Check(ent, ce)
	}
	return ce
}

func (mc multiCore) Write(ent Entry, fields []Field) error {
	var err error
	for i := range mc {
		err = multierr.Append(err, mc[i].Write(ent, fields))
	}
	return err
}

func (mc multiCore) Sync() error {
	var err error
	for i := range mc {
		err = multierr.Append(err, mc[i].Sync())
	}
	return err
}
