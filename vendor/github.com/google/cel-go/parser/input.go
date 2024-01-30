// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parser

import (
	antlr "github.com/antlr/antlr4/runtime/Go/antlr/v4"

	"github.com/google/cel-go/common/runes"
)

type charStream struct {
	buf runes.Buffer
	pos int
	src string
}

// Consume implements (antlr.CharStream).Consume.
func (c *charStream) Consume() {
	if c.pos >= c.buf.Len() {
		panic("cannot consume EOF")
	}
	c.pos++
}

// LA implements (antlr.CharStream).LA.
func (c *charStream) LA(offset int) int {
	if offset == 0 {
		return 0
	}
	if offset < 0 {
		offset++
	}
	pos := c.pos + offset - 1
	if pos < 0 || pos >= c.buf.Len() {
		return antlr.TokenEOF
	}
	return int(c.buf.Get(pos))
}

// LT mimics (*antlr.InputStream).LT.
func (c *charStream) LT(offset int) int {
	return c.LA(offset)
}

// Mark implements (antlr.CharStream).Mark.
func (c *charStream) Mark() int {
	return -1
}

// Release implements (antlr.CharStream).Release.
func (c *charStream) Release(marker int) {}

// Index implements (antlr.CharStream).Index.
func (c *charStream) Index() int {
	return c.pos
}

// Seek implements (antlr.CharStream).Seek.
func (c *charStream) Seek(index int) {
	if index <= c.pos {
		c.pos = index
		return
	}
	if index < c.buf.Len() {
		c.pos = index
	} else {
		c.pos = c.buf.Len()
	}
}

// Size implements (antlr.CharStream).Size.
func (c *charStream) Size() int {
	return c.buf.Len()
}

// GetSourceName implements (antlr.CharStream).GetSourceName.
func (c *charStream) GetSourceName() string {
	return c.src
}

// GetText implements (antlr.CharStream).GetText.
func (c *charStream) GetText(start, stop int) string {
	if stop >= c.buf.Len() {
		stop = c.buf.Len() - 1
	}
	if start >= c.buf.Len() {
		return ""
	}
	return c.buf.Slice(start, stop+1)
}

// GetTextFromTokens implements (antlr.CharStream).GetTextFromTokens.
func (c *charStream) GetTextFromTokens(start, stop antlr.Token) string {
	if start != nil && stop != nil {
		return c.GetText(start.GetTokenIndex(), stop.GetTokenIndex())
	}
	return ""
}

// GetTextFromInterval implements (antlr.CharStream).GetTextFromInterval.
func (c *charStream) GetTextFromInterval(i *antlr.Interval) string {
	return c.GetText(i.Start, i.Stop)
}

// String mimics (*antlr.InputStream).String.
func (c *charStream) String() string {
	return c.buf.Slice(0, c.buf.Len())
}

var _ antlr.CharStream = &charStream{}

func newCharStream(buf runes.Buffer, desc string) antlr.CharStream {
	return &charStream{
		buf: buf,
		src: desc,
	}
}
