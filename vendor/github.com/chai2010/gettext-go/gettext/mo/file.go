// Copyright 2013 ChaiShushan <chaishushan{AT}gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mo

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"strings"
)

const (
	MoHeaderSize        = 28
	MoMagicLittleEndian = 0x950412de
	MoMagicBigEndian    = 0xde120495

	EotSeparator = "\x04" // msgctxt and msgid separator
	NulSeparator = "\x00" // msgid and msgstr separator
)

// File represents an MO File.
//
// See http://www.gnu.org/software/gettext/manual/html_node/MO-Files.html
type File struct {
	MagicNumber  uint32
	MajorVersion uint16
	MinorVersion uint16
	MsgIdCount   uint32
	MsgIdOffset  uint32
	MsgStrOffset uint32
	HashSize     uint32
	HashOffset   uint32
	MimeHeader   Header
	Messages     []Message
}

// Load loads a named mo file.
func Load(name string) (*File, error) {
	data, err := ioutil.ReadFile(name)
	if err != nil {
		return nil, err
	}
	return LoadData(data)
}

// LoadData loads mo file format data.
func LoadData(data []byte) (*File, error) {
	r := bytes.NewReader(data)

	var magicNumber uint32
	if err := binary.Read(r, binary.LittleEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("gettext: %v", err)
	}
	var bo binary.ByteOrder
	switch magicNumber {
	case MoMagicLittleEndian:
		bo = binary.LittleEndian
	case MoMagicBigEndian:
		bo = binary.BigEndian
	default:
		return nil, fmt.Errorf("gettext: %v", "invalid magic number")
	}

	var header struct {
		MajorVersion uint16
		MinorVersion uint16
		MsgIdCount   uint32
		MsgIdOffset  uint32
		MsgStrOffset uint32
		HashSize     uint32
		HashOffset   uint32
	}
	if err := binary.Read(r, bo, &header); err != nil {
		return nil, fmt.Errorf("gettext: %v", err)
	}
	if v := header.MajorVersion; v != 0 && v != 1 {
		return nil, fmt.Errorf("gettext: %v", "invalid version number")
	}
	if v := header.MinorVersion; v != 0 && v != 1 {
		return nil, fmt.Errorf("gettext: %v", "invalid version number")
	}

	msgIdStart := make([]uint32, header.MsgIdCount)
	msgIdLen := make([]uint32, header.MsgIdCount)
	if _, err := r.Seek(int64(header.MsgIdOffset), 0); err != nil {
		return nil, fmt.Errorf("gettext: %v", err)
	}
	for i := 0; i < int(header.MsgIdCount); i++ {
		if err := binary.Read(r, bo, &msgIdLen[i]); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
		if err := binary.Read(r, bo, &msgIdStart[i]); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
	}

	msgStrStart := make([]int32, header.MsgIdCount)
	msgStrLen := make([]int32, header.MsgIdCount)
	if _, err := r.Seek(int64(header.MsgStrOffset), 0); err != nil {
		return nil, fmt.Errorf("gettext: %v", err)
	}
	for i := 0; i < int(header.MsgIdCount); i++ {
		if err := binary.Read(r, bo, &msgStrLen[i]); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
		if err := binary.Read(r, bo, &msgStrStart[i]); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
	}

	file := &File{
		MagicNumber:  magicNumber,
		MajorVersion: header.MajorVersion,
		MinorVersion: header.MinorVersion,
		MsgIdCount:   header.MsgIdCount,
		MsgIdOffset:  header.MsgIdOffset,
		MsgStrOffset: header.MsgStrOffset,
		HashSize:     header.HashSize,
		HashOffset:   header.HashOffset,
	}
	for i := 0; i < int(header.MsgIdCount); i++ {
		if _, err := r.Seek(int64(msgIdStart[i]), 0); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
		msgIdData := make([]byte, msgIdLen[i])
		if _, err := r.Read(msgIdData); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}

		if _, err := r.Seek(int64(msgStrStart[i]), 0); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}
		msgStrData := make([]byte, msgStrLen[i])
		if _, err := r.Read(msgStrData); err != nil {
			return nil, fmt.Errorf("gettext: %v", err)
		}

		if len(msgIdData) == 0 {
			var msg = Message{
				MsgId:  string(msgIdData),
				MsgStr: string(msgStrData),
			}
			file.MimeHeader.fromMessage(&msg)
		} else {
			var msg = Message{
				MsgId:  string(msgIdData),
				MsgStr: string(msgStrData),
			}
			// Is this a context message?
			if idx := strings.Index(msg.MsgId, EotSeparator); idx != -1 {
				msg.MsgContext, msg.MsgId = msg.MsgId[:idx], msg.MsgId[idx+1:]
			}
			// Is this a plural message?
			if idx := strings.Index(msg.MsgId, NulSeparator); idx != -1 {
				msg.MsgId, msg.MsgIdPlural = msg.MsgId[:idx], msg.MsgId[idx+1:]
				msg.MsgStrPlural = strings.Split(msg.MsgStr, NulSeparator)
				msg.MsgStr = ""
			}
			file.Messages = append(file.Messages, msg)
		}
	}

	return file, nil
}

// Save saves a mo file.
func (f *File) Save(name string) error {
	return ioutil.WriteFile(name, f.Data(), 0666)
}

// Save returns a mo file format data.
func (f *File) Data() []byte {
	return encodeFile(f)
}

// String returns the po format file string.
func (f *File) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "# version: %d.%d\n", f.MajorVersion, f.MinorVersion)
	fmt.Fprintf(&buf, "%s\n", f.MimeHeader.String())
	fmt.Fprintf(&buf, "\n")

	for k, v := range f.Messages {
		fmt.Fprintf(&buf, `msgid "%v"`+"\n", k)
		fmt.Fprintf(&buf, `msgstr "%s"`+"\n", v.MsgStr)
		fmt.Fprintf(&buf, "\n")
	}

	return buf.String()
}
