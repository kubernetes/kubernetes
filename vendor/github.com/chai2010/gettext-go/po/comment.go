// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package po

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Comment represents every message's comments.
type Comment struct {
	StartLine         int      // comment start line
	TranslatorComment string   // #  translator-comments // TrimSpace
	ExtractedComment  string   // #. extracted-comments
	ReferenceFile     []string // #: src/msgcmp.c:338 src/po-lex.c:699
	ReferenceLine     []int    // #: src/msgcmp.c:338 src/po-lex.c:699
	Flags             []string // #, fuzzy,c-format,range:0..10
	PrevMsgContext    string   // #| msgctxt previous-context
	PrevMsgId         string   // #| msgid previous-untranslated-string
}

func (p *Comment) less(q *Comment) bool {
	if p.StartLine != 0 || q.StartLine != 0 {
		return p.StartLine < q.StartLine
	}
	if a, b := len(p.ReferenceFile), len(q.ReferenceFile); a != b {
		return a < b
	}
	for i := 0; i < len(p.ReferenceFile); i++ {
		if a, b := p.ReferenceFile[i], q.ReferenceFile[i]; a != b {
			return a < b
		}
		if a, b := p.ReferenceLine[i], q.ReferenceLine[i]; a != b {
			return a < b
		}
	}
	return false
}

func (p *Comment) readPoComment(r *lineReader) (err error) {
	*p = Comment{}
	if err = r.skipBlankLine(); err != nil {
		return err
	}
	defer func(oldPos int) {
		newPos := r.currentPos()
		if newPos != oldPos && err == io.EOF {
			err = nil
		}
	}(r.currentPos())

	p.StartLine = r.currentPos() + 1
	for {
		var s string
		if s, _, err = r.currentLine(); err != nil {
			return
		}
		if len(s) == 0 || s[0] != '#' {
			return
		}

		if err = p.readTranslatorComment(r); err != nil {
			return
		}
		if err = p.readExtractedComment(r); err != nil {
			return
		}
		if err = p.readReferenceComment(r); err != nil {
			return
		}
		if err = p.readFlagsComment(r); err != nil {
			return
		}
		if err = p.readPrevMsgContext(r); err != nil {
			return
		}
		if err = p.readPrevMsgId(r); err != nil {
			return
		}
	}
}

func (p *Comment) readTranslatorComment(r *lineReader) (err error) {
	const prefix = "# " // .,:|
	for {
		var s string
		if s, _, err = r.readLine(); err != nil {
			return err
		}
		if len(s) < 1 || s[0] != '#' {
			r.unreadLine()
			return nil
		}
		if len(s) >= 2 {
			switch s[1] {
			case '.', ',', ':', '|':
				r.unreadLine()
				return nil
			}
		}
		if p.TranslatorComment != "" {
			p.TranslatorComment += "\n"
		}
		p.TranslatorComment += strings.TrimSpace(s[1:])
	}
}

func (p *Comment) readExtractedComment(r *lineReader) (err error) {
	const prefix = "#."
	for {
		var s string
		if s, _, err = r.readLine(); err != nil {
			return err
		}
		if len(s) < len(prefix) || s[:len(prefix)] != prefix {
			r.unreadLine()
			return nil
		}
		if p.ExtractedComment != "" {
			p.ExtractedComment += "\n"
		}
		p.ExtractedComment += strings.TrimSpace(s[len(prefix):])
	}
}

func (p *Comment) readReferenceComment(r *lineReader) (err error) {
	const prefix = "#:"
	for {
		var s string
		if s, _, err = r.readLine(); err != nil {
			return err
		}
		if len(s) < len(prefix) || s[:len(prefix)] != prefix {
			r.unreadLine()
			return nil
		}
		ss := strings.Split(strings.TrimSpace(s[len(prefix):]), " ")
		for i := 0; i < len(ss); i++ {
			idx := strings.Index(ss[i], ":")
			if idx <= 0 {
				continue
			}
			name := strings.TrimSpace(ss[i][:idx])
			line, _ := strconv.Atoi(strings.TrimSpace(ss[i][idx+1:]))
			p.ReferenceFile = append(p.ReferenceFile, name)
			p.ReferenceLine = append(p.ReferenceLine, line)
		}
	}
}

func (p *Comment) readFlagsComment(r *lineReader) (err error) {
	const prefix = "#,"
	for {
		var s string
		if s, _, err = r.readLine(); err != nil {
			return err
		}
		if len(s) < len(prefix) || s[:len(prefix)] != prefix {
			r.unreadLine()
			return nil
		}
		ss := strings.Split(strings.TrimSpace(s[len(prefix):]), ",")
		for i := 0; i < len(ss); i++ {
			p.Flags = append(p.Flags, strings.TrimSpace(ss[i]))
		}
	}
}

func (p *Comment) readPrevMsgContext(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !rePrevMsgContextComments.MatchString(s) {
		return
	}
	p.PrevMsgContext, err = p.readString(r)
	return
}

func (p *Comment) readPrevMsgId(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !rePrevMsgIdComments.MatchString(s) {
		return
	}
	p.PrevMsgId, err = p.readString(r)
	return
}

func (p *Comment) readString(r *lineReader) (msg string, err error) {
	var s string
	if s, _, err = r.readLine(); err != nil {
		return
	}
	msg += decodePoString(s)
	for {
		if s, _, err = r.readLine(); err != nil {
			return
		}
		if !reStringLineComments.MatchString(s) {
			r.unreadLine()
			break
		}
		msg += decodePoString(s)
	}
	return
}

// GetFuzzy gets the fuzzy flag.
func (p *Comment) GetFuzzy() bool {
	for _, s := range p.Flags {
		if s == "fuzzy" {
			return true
		}
	}
	return false
}

// SetFuzzy sets the fuzzy flag.
func (p *Comment) SetFuzzy(fuzzy bool) {
	//
}

// String returns the po format comment string.
func (p Comment) String() string {
	var buf bytes.Buffer
	if p.TranslatorComment != "" {
		ss := strings.Split(p.TranslatorComment, "\n")
		for i := 0; i < len(ss); i++ {
			fmt.Fprintf(&buf, "# %s\n", ss[i])
		}
	}
	if p.ExtractedComment != "" {
		ss := strings.Split(p.ExtractedComment, "\n")
		for i := 0; i < len(ss); i++ {
			fmt.Fprintf(&buf, "#. %s\n", ss[i])
		}
	}
	if a, b := len(p.ReferenceFile), len(p.ReferenceLine); a != 0 && a == b {
		fmt.Fprintf(&buf, "#:")
		for i := 0; i < len(p.ReferenceFile); i++ {
			fmt.Fprintf(&buf, " %s:%d", p.ReferenceFile[i], p.ReferenceLine[i])
		}
		fmt.Fprintf(&buf, "\n")
	}
	if len(p.Flags) != 0 {
		fmt.Fprintf(&buf, "#, %s", p.Flags[0])
		for i := 1; i < len(p.Flags); i++ {
			fmt.Fprintf(&buf, ", %s", p.Flags[i])
		}
		fmt.Fprintf(&buf, "\n")
	}
	if p.PrevMsgContext != "" {
		s := encodeCommentPoString(p.PrevMsgContext)
		fmt.Fprintf(&buf, "#| msgctxt %s\n", s)
	}
	if p.PrevMsgId != "" {
		s := encodeCommentPoString(p.PrevMsgId)
		fmt.Fprintf(&buf, "#| msgid %s\n", s)
	}
	return buf.String()
}
