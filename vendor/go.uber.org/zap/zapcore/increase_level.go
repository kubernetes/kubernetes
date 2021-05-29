// Copyright (c) 2020 Uber Technologies, Inc.
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

import "fmt"

type levelFilterCore struct {
	core  Core
	level LevelEnabler
}

// NewIncreaseLevelCore creates a core that can be used to increase the level of
// an existing Core. It cannot be used to decrease the logging level, as it acts
// as a filter before calling the underlying core. If level decreases the log level,
// an error is returned.
func NewIncreaseLevelCore(core Core, level LevelEnabler) (Core, error) {
	for l := _maxLevel; l >= _minLevel; l-- {
		if !core.Enabled(l) && level.Enabled(l) {
			return nil, fmt.Errorf("invalid increase level, as level %q is allowed by increased level, but not by existing core", l)
		}
	}

	return &levelFilterCore{core, level}, nil
}

func (c *levelFilterCore) Enabled(lvl Level) bool {
	return c.level.Enabled(lvl)
}

func (c *levelFilterCore) With(fields []Field) Core {
	return &levelFilterCore{c.core.With(fields), c.level}
}

func (c *levelFilterCore) Check(ent Entry, ce *CheckedEntry) *CheckedEntry {
	if !c.Enabled(ent.Level) {
		return ce
	}

	return c.core.Check(ent, ce)
}

func (c *levelFilterCore) Write(ent Entry, fields []Field) error {
	return c.core.Write(ent, fields)
}

func (c *levelFilterCore) Sync() error {
	return c.core.Sync()
}
