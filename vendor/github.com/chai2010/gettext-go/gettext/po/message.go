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

// A PO file is made up of many entries,
// each entry holding the relation between an original untranslated string
// and its corresponding translation.
//
// See http://www.gnu.org/software/gettext/manual/html_node/PO-Files.html
type Message struct {
	Comment               // Coments
	MsgContext   string   // msgctxt context
	MsgId        string   // msgid untranslated-string
	MsgIdPlural  string   // msgid_plural untranslated-string-plural
	MsgStr       string   // msgstr translated-string
	MsgStrPlural []string // msgstr[0] translated-string-case-0
}

type byMessages []Message

func (d byMessages) Len() int {
	return len(d)
}
func (d byMessages) Less(i, j int) bool {
	if d[i].Comment.less(&d[j].Comment) {
		return true
	}
	if a, b := d[i].MsgContext, d[j].MsgContext; a != b {
		return a < b
	}
	if a, b := d[i].MsgId, d[j].MsgId; a != b {
		return a < b
	}
	if a, b := d[i].MsgIdPlural, d[j].MsgIdPlural; a != b {
		return a < b
	}
	return false
}
func (d byMessages) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}

func (p *Message) readPoEntry(r *lineReader) (err error) {
	*p = Message{}
	if err = r.skipBlankLine(); err != nil {
		return
	}
	defer func(oldPos int) {
		newPos := r.currentPos()
		if newPos != oldPos && err == io.EOF {
			err = nil
		}
	}(r.currentPos())

	if err = p.Comment.readPoComment(r); err != nil {
		return
	}
	for {
		var s string
		if s, _, err = r.currentLine(); err != nil {
			return
		}

		if p.isInvalidLine(s) {
			err = fmt.Errorf("gettext: line %d, %v", r.currentPos(), "invalid line")
			return
		}
		if reComment.MatchString(s) || reBlankLine.MatchString(s) {
			return
		}

		if err = p.readMsgContext(r); err != nil {
			return
		}
		if err = p.readMsgId(r); err != nil {
			return
		}
		if err = p.readMsgIdPlural(r); err != nil {
			return
		}
		if err = p.readMsgStrOrPlural(r); err != nil {
			return
		}
	}
}

func (p *Message) readMsgContext(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !reMsgContext.MatchString(s) {
		return
	}
	p.MsgContext, err = p.readString(r)
	return
}

func (p *Message) readMsgId(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !reMsgId.MatchString(s) {
		return
	}
	p.MsgId, err = p.readString(r)
	return
}

func (p *Message) readMsgIdPlural(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !reMsgIdPlural.MatchString(s) {
		return
	}
	p.MsgIdPlural, err = p.readString(r)
	return nil
}

func (p *Message) readMsgStrOrPlural(r *lineReader) (err error) {
	var s string
	if s, _, err = r.currentLine(); err != nil {
		return
	}
	if !reMsgStr.MatchString(s) && !reMsgStrPlural.MatchString(s) {
		return
	}
	if reMsgStrPlural.MatchString(s) {
		left, right := strings.Index(s, `[`), strings.LastIndex(s, `]`)
		idx, _ := strconv.Atoi(s[left+1 : right])
		s, err = p.readString(r)
		if n := len(p.MsgStrPlural); (idx + 1) > n {
			p.MsgStrPlural = append(p.MsgStrPlural, make([]string, (idx+1)-n)...)
		}
		p.MsgStrPlural[idx] = s
	} else {
		p.MsgStr, err = p.readString(r)
	}
	return nil
}

func (p *Message) readString(r *lineReader) (msg string, err error) {
	var s string
	if s, _, err = r.readLine(); err != nil {
		return
	}
	msg += decodePoString(s)
	for {
		if s, _, err = r.readLine(); err != nil {
			return
		}
		if !reStringLine.MatchString(s) {
			r.unreadLine()
			break
		}
		msg += decodePoString(s)
	}
	return
}

// String returns the po format entry string.
func (p Message) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s", p.Comment.String())
	fmt.Fprintf(&buf, "msgid %s", encodePoString(p.MsgId))
	if p.MsgIdPlural != "" {
		fmt.Fprintf(&buf, "msgid_plural %s", encodePoString(p.MsgIdPlural))
	}
	if p.MsgStr != "" {
		fmt.Fprintf(&buf, "msgstr %s", encodePoString(p.MsgStr))
	}
	for i := 0; i < len(p.MsgStrPlural); i++ {
		fmt.Fprintf(&buf, "msgstr[%d] %s", i, encodePoString(p.MsgStrPlural[i]))
	}
	return buf.String()
}
