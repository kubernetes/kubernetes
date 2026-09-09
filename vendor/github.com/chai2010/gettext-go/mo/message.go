// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mo

import (
	"bytes"
	"fmt"
)

// A MO file is made up of many entries,
// each entry holding the relation between an original untranslated string
// and its corresponding translation.
//
// See http://www.gnu.org/software/gettext/manual/html_node/MO-Files.html
type Message struct {
	MsgContext   string   // msgctxt context
	MsgId        string   // msgid untranslated-string
	MsgIdPlural  string   // msgid_plural untranslated-string-plural
	MsgStr       string   // msgstr translated-string
	MsgStrPlural []string // msgstr[0] translated-string-case-0
}

// String returns the po format entry string.
func (p Message) String() string {
	var buf bytes.Buffer
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

func (m_i *Message) less(m_j *Message) bool {
	if a, b := m_i.MsgContext, m_j.MsgContext; a != b {
		return a < b
	}
	if a, b := m_i.MsgId, m_j.MsgId; a != b {
		return a < b
	}
	if a, b := m_i.MsgIdPlural, m_j.MsgIdPlural; a != b {
		return a < b
	}
	return false
}
