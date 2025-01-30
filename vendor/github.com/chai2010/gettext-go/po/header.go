// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package po

import (
	"bytes"
	"fmt"
	"strings"
)

// Header is the initial comments "SOME DESCRIPTIVE TITLE", "YEAR"
// and "FIRST AUTHOR <EMAIL@ADDRESS>, YEAR" ought to be replaced by sensible information.
//
// See http://www.gnu.org/software/gettext/manual/html_node/Header-Entry.html#Header-Entry
type Header struct {
	Comment                        // Header Comments
	ProjectIdVersion        string // Project-Id-Version: PACKAGE VERSION
	ReportMsgidBugsTo       string // Report-Msgid-Bugs-To: FIRST AUTHOR <EMAIL@ADDRESS>
	POTCreationDate         string // POT-Creation-Date: YEAR-MO-DA HO:MI+ZONE
	PORevisionDate          string // PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
	LastTranslator          string // Last-Translator: FIRST AUTHOR <EMAIL@ADDRESS>
	LanguageTeam            string // Language-Team: golang-china
	Language                string // Language: zh_CN
	MimeVersion             string // MIME-Version: 1.0
	ContentType             string // Content-Type: text/plain; charset=UTF-8
	ContentTransferEncoding string // Content-Transfer-Encoding: 8bit
	PluralForms             string // Plural-Forms: nplurals=2; plural=n == 1 ? 0 : 1;
	XGenerator              string // X-Generator: Poedit 1.5.5
	UnknowFields            map[string]string
}

func (p *Header) parseHeader(msg *Message) {
	if msg.MsgId != "" || msg.MsgStr == "" {
		return
	}
	lines := strings.Split(msg.MsgStr, "\n")
	for i := 0; i < len(lines); i++ {
		idx := strings.Index(lines[i], ":")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(lines[i][:idx])
		val := strings.TrimSpace(lines[i][idx+1:])
		switch strings.ToUpper(key) {
		case strings.ToUpper("Project-Id-Version"):
			p.ProjectIdVersion = val
		case strings.ToUpper("Report-Msgid-Bugs-To"):
			p.ReportMsgidBugsTo = val
		case strings.ToUpper("POT-Creation-Date"):
			p.POTCreationDate = val
		case strings.ToUpper("PO-Revision-Date"):
			p.PORevisionDate = val
		case strings.ToUpper("Last-Translator"):
			p.LastTranslator = val
		case strings.ToUpper("Language-Team"):
			p.LanguageTeam = val
		case strings.ToUpper("Language"):
			p.Language = val
		case strings.ToUpper("MIME-Version"):
			p.MimeVersion = val
		case strings.ToUpper("Content-Type"):
			p.ContentType = val
		case strings.ToUpper("Content-Transfer-Encoding"):
			p.ContentTransferEncoding = val
		case strings.ToUpper("Plural-Forms"):
			p.PluralForms = val
		case strings.ToUpper("X-Generator"):
			p.XGenerator = val
		default:
			if p.UnknowFields == nil {
				p.UnknowFields = make(map[string]string)
			}
			p.UnknowFields[key] = val
		}
	}
	p.Comment = msg.Comment
}

// String returns the po format header string.
func (p Header) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s", p.Comment.String())
	fmt.Fprintf(&buf, `msgid ""`+"\n")
	fmt.Fprintf(&buf, `msgstr ""`+"\n")
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Project-Id-Version", p.ProjectIdVersion)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Report-Msgid-Bugs-To", p.ReportMsgidBugsTo)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "POT-Creation-Date", p.POTCreationDate)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "PO-Revision-Date", p.PORevisionDate)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Last-Translator", p.LastTranslator)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Language-Team", p.LanguageTeam)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Language", p.Language)
	if p.MimeVersion != "" {
		fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "MIME-Version", p.MimeVersion)
	}
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Content-Type", p.ContentType)
	fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "Content-Transfer-Encoding", p.ContentTransferEncoding)
	if p.XGenerator != "" {
		fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", "X-Generator", p.XGenerator)
	}
	for k, v := range p.UnknowFields {
		fmt.Fprintf(&buf, `"%s: %s\n"`+"\n", k, v)
	}
	return buf.String()
}
