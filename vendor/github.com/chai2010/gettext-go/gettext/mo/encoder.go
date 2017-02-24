// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mo

import (
	"bytes"
	"encoding/binary"
	"sort"
	"strings"
)

type moHeader struct {
	MagicNumber  uint32
	MajorVersion uint16
	MinorVersion uint16
	MsgIdCount   uint32
	MsgIdOffset  uint32
	MsgStrOffset uint32
	HashSize     uint32
	HashOffset   uint32
}

type moStrPos struct {
	Size uint32 // must keep fields order
	Addr uint32
}

func encodeFile(f *File) []byte {
	hdr := &moHeader{
		MagicNumber: MoMagicLittleEndian,
	}
	data := encodeData(hdr, f)
	data = append(encodeHeader(hdr), data...)
	return data
}

// encode data and init moHeader
func encodeData(hdr *moHeader, f *File) []byte {
	msgList := []Message{f.MimeHeader.toMessage()}
	for _, v := range f.Messages {
		if len(v.MsgId) == 0 {
			continue
		}
		if len(v.MsgStr) == 0 && len(v.MsgStrPlural) == 0 {
			continue
		}
		msgList = append(msgList, v)
	}
	sort.Sort(byMessages(msgList))

	var buf bytes.Buffer
	var msgIdPosList = make([]moStrPos, len(msgList))
	var msgStrPosList = make([]moStrPos, len(msgList))
	for i, v := range msgList {
		// write msgid
		msgId := encodeMsgId(v)
		msgIdPosList[i].Addr = uint32(buf.Len() + MoHeaderSize)
		msgIdPosList[i].Size = uint32(len(msgId))
		buf.WriteString(msgId)
		// write msgstr
		msgStr := encodeMsgStr(v)
		msgStrPosList[i].Addr = uint32(buf.Len() + MoHeaderSize)
		msgStrPosList[i].Size = uint32(len(msgStr))
		buf.WriteString(msgStr)
	}

	hdr.MsgIdOffset = uint32(buf.Len() + MoHeaderSize)
	binary.Write(&buf, binary.LittleEndian, msgIdPosList)
	hdr.MsgStrOffset = uint32(buf.Len() + MoHeaderSize)
	binary.Write(&buf, binary.LittleEndian, msgStrPosList)

	hdr.MsgIdCount = uint32(len(msgList))
	return buf.Bytes()
}

// must called after encodeData
func encodeHeader(hdr *moHeader) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, hdr)
	return buf.Bytes()
}

func encodeMsgId(v Message) string {
	if v.MsgContext != "" && v.MsgIdPlural != "" {
		return v.MsgContext + EotSeparator + v.MsgId + NulSeparator + v.MsgIdPlural
	}
	if v.MsgContext != "" && v.MsgIdPlural == "" {
		return v.MsgContext + EotSeparator + v.MsgId
	}
	if v.MsgContext == "" && v.MsgIdPlural != "" {
		return v.MsgId + NulSeparator + v.MsgIdPlural
	}
	return v.MsgId
}

func encodeMsgStr(v Message) string {
	if v.MsgIdPlural != "" {
		return strings.Join(v.MsgStrPlural, NulSeparator)
	}
	return v.MsgStr
}

type byMessages []Message

func (d byMessages) Len() int {
	return len(d)
}
func (d byMessages) Less(i, j int) bool {
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
