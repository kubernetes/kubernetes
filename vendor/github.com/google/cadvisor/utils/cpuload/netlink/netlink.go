// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package netlink

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"syscall"

	info "github.com/google/cadvisor/info/v1"
	"golang.org/x/sys/unix"
)

var (
	// TODO(rjnagal): Verify and fix for other architectures.
	Endian = binary.LittleEndian
)

type genMsghdr struct {
	Command  uint8
	Version  uint8
	Reserved uint16
}

type netlinkMessage struct {
	Header    syscall.NlMsghdr
	GenHeader genMsghdr
	Data      []byte
}

func (self netlinkMessage) toRawMsg() (rawmsg syscall.NetlinkMessage) {
	rawmsg.Header = self.Header
	w := bytes.NewBuffer([]byte{})
	binary.Write(w, Endian, self.GenHeader)
	w.Write(self.Data)
	rawmsg.Data = w.Bytes()
	return rawmsg
}

type loadStatsResp struct {
	Header    syscall.NlMsghdr
	GenHeader genMsghdr
	Stats     info.LoadStats
}

// Return required padding to align 'size' to 'alignment'.
func padding(size int, alignment int) int {
	unalignedPart := size % alignment
	return (alignment - unalignedPart) % alignment
}

// Get family id for taskstats subsystem.
func getFamilyId(conn *Connection) (uint16, error) {
	msg := prepareFamilyMessage()
	conn.WriteMessage(msg.toRawMsg())

	resp, err := conn.ReadMessage()
	if err != nil {
		return 0, err
	}
	id, err := parseFamilyResp(resp)
	if err != nil {
		return 0, err
	}
	return id, nil
}

// Append an attribute to the message.
// Adds attribute info (length and type), followed by the data and necessary padding.
// Can be called multiple times to add attributes. Only fixed size and string type
// attributes are handled. We don't need nested attributes for task stats.
func addAttribute(buf *bytes.Buffer, attrType uint16, data interface{}, dataSize int) {
	attr := syscall.RtAttr{
		Len:  syscall.SizeofRtAttr,
		Type: attrType,
	}
	attr.Len += uint16(dataSize)
	binary.Write(buf, Endian, attr)
	switch data := data.(type) {
	case string:
		binary.Write(buf, Endian, []byte(data))
		buf.WriteByte(0) // terminate
	default:
		binary.Write(buf, Endian, data)
	}
	for i := 0; i < padding(int(attr.Len), syscall.NLMSG_ALIGNTO); i++ {
		buf.WriteByte(0)
	}
}

// Prepares the message and generic headers and appends attributes as data.
func prepareMessage(headerType uint16, cmd uint8, attributes []byte) (msg netlinkMessage) {
	msg.Header.Type = headerType
	msg.Header.Flags = syscall.NLM_F_REQUEST
	msg.GenHeader.Command = cmd
	msg.GenHeader.Version = 0x1
	msg.Data = attributes
	return msg
}

// Prepares message to query family id for task stats.
func prepareFamilyMessage() (msg netlinkMessage) {
	buf := bytes.NewBuffer([]byte{})
	addAttribute(buf, unix.CTRL_ATTR_FAMILY_NAME, unix.TASKSTATS_GENL_NAME, len(unix.TASKSTATS_GENL_NAME)+1)
	return prepareMessage(unix.GENL_ID_CTRL, unix.CTRL_CMD_GETFAMILY, buf.Bytes())
}

// Prepares message to query task stats for a task group.
func prepareCmdMessage(id uint16, cfd uintptr) (msg netlinkMessage) {
	buf := bytes.NewBuffer([]byte{})
	addAttribute(buf, unix.CGROUPSTATS_CMD_ATTR_FD, uint32(cfd), 4)
	return prepareMessage(id, unix.CGROUPSTATS_CMD_GET, buf.Bytes())
}

// Extracts returned family id from the response.
func parseFamilyResp(msg syscall.NetlinkMessage) (uint16, error) {
	m := new(netlinkMessage)
	m.Header = msg.Header
	err := verifyHeader(msg)
	if err != nil {
		return 0, err
	}
	buf := bytes.NewBuffer(msg.Data)
	// extract generic header from data.
	err = binary.Read(buf, Endian, &m.GenHeader)
	if err != nil {
		return 0, err
	}
	id := uint16(0)
	// Extract attributes. kernel reports family name, id, version, etc.
	// Scan till we find id.
	for buf.Len() > syscall.SizeofRtAttr {
		var attr syscall.RtAttr
		err = binary.Read(buf, Endian, &attr)
		if err != nil {
			return 0, err
		}
		if attr.Type == unix.CTRL_ATTR_FAMILY_ID {
			err = binary.Read(buf, Endian, &id)
			if err != nil {
				return 0, err
			}
			return id, nil
		}
		payload := int(attr.Len) - syscall.SizeofRtAttr
		skipLen := payload + padding(payload, syscall.SizeofRtAttr)
		name := make([]byte, skipLen)
		err = binary.Read(buf, Endian, name)
		if err != nil {
			return 0, err
		}
	}
	return 0, fmt.Errorf("family id not found in the response.")
}

// Extract task stats from response returned by kernel.
func parseLoadStatsResp(msg syscall.NetlinkMessage) (*loadStatsResp, error) {
	m := new(loadStatsResp)
	m.Header = msg.Header
	err := verifyHeader(msg)
	if err != nil {
		return m, err
	}
	buf := bytes.NewBuffer(msg.Data)
	// Scan the general header.
	err = binary.Read(buf, Endian, &m.GenHeader)
	if err != nil {
		return m, err
	}
	// cgroup stats response should have just one attribute.
	// Read it directly into the stats structure.
	var attr syscall.RtAttr
	err = binary.Read(buf, Endian, &attr)
	if err != nil {
		return m, err
	}
	err = binary.Read(buf, Endian, &m.Stats)
	if err != nil {
		return m, err
	}
	return m, err
}

// Verify and return any error reported by kernel.
func verifyHeader(msg syscall.NetlinkMessage) error {
	switch msg.Header.Type {
	case syscall.NLMSG_DONE:
		return fmt.Errorf("expected a response, got nil")
	case syscall.NLMSG_ERROR:
		buf := bytes.NewBuffer(msg.Data)
		var errno int32
		binary.Read(buf, Endian, errno)
		return fmt.Errorf("netlink request failed with error %s", syscall.Errno(-errno))
	}
	return nil
}

// Get load stats for a task group.
// id: family id for taskstats.
// cfd: open file to path to the cgroup directory under cpu hierarchy.
// conn: open netlink connection used to communicate with kernel.
func getLoadStats(id uint16, cfd *os.File, conn *Connection) (info.LoadStats, error) {
	msg := prepareCmdMessage(id, cfd.Fd())
	err := conn.WriteMessage(msg.toRawMsg())
	if err != nil {
		return info.LoadStats{}, err
	}

	resp, err := conn.ReadMessage()
	if err != nil {
		return info.LoadStats{}, err
	}

	parsedmsg, err := parseLoadStatsResp(resp)
	if err != nil {
		return info.LoadStats{}, err
	}
	return parsedmsg.Stats, nil
}
