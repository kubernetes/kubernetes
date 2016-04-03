// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"golang.org/x/crypto/openpgp/errors"
	"io"
	"io/ioutil"
)

// OpaquePacket represents an OpenPGP packet as raw, unparsed data. This is
// useful for splitting and storing the original packet contents separately,
// handling unsupported packet types or accessing parts of the packet not yet
// implemented by this package.
type OpaquePacket struct {
	// Packet type
	Tag uint8
	// Reason why the packet was parsed opaquely
	Reason error
	// Binary contents of the packet data
	Contents []byte
}

func (op *OpaquePacket) parse(r io.Reader) (err error) {
	op.Contents, err = ioutil.ReadAll(r)
	return
}

// Serialize marshals the packet to a writer in its original form, including
// the packet header.
func (op *OpaquePacket) Serialize(w io.Writer) (err error) {
	err = serializeHeader(w, packetType(op.Tag), len(op.Contents))
	if err == nil {
		_, err = w.Write(op.Contents)
	}
	return
}

// Parse attempts to parse the opaque contents into a structure supported by
// this package. If the packet is not known then the result will be another
// OpaquePacket.
func (op *OpaquePacket) Parse() (p Packet, err error) {
	hdr := bytes.NewBuffer(nil)
	err = serializeHeader(hdr, packetType(op.Tag), len(op.Contents))
	if err != nil {
		op.Reason = err
		return op, err
	}
	p, err = Read(io.MultiReader(hdr, bytes.NewBuffer(op.Contents)))
	if err != nil {
		op.Reason = err
		p = op
	}
	return
}

// OpaqueReader reads OpaquePackets from an io.Reader.
type OpaqueReader struct {
	r io.Reader
}

func NewOpaqueReader(r io.Reader) *OpaqueReader {
	return &OpaqueReader{r: r}
}

// Read the next OpaquePacket.
func (or *OpaqueReader) Next() (op *OpaquePacket, err error) {
	tag, _, contents, err := readHeader(or.r)
	if err != nil {
		return
	}
	op = &OpaquePacket{Tag: uint8(tag), Reason: err}
	err = op.parse(contents)
	if err != nil {
		consumeAll(contents)
	}
	return
}

// OpaqueSubpacket represents an unparsed OpenPGP subpacket,
// as found in signature and user attribute packets.
type OpaqueSubpacket struct {
	SubType  uint8
	Contents []byte
}

// OpaqueSubpackets extracts opaque, unparsed OpenPGP subpackets from
// their byte representation.
func OpaqueSubpackets(contents []byte) (result []*OpaqueSubpacket, err error) {
	var (
		subHeaderLen int
		subPacket    *OpaqueSubpacket
	)
	for len(contents) > 0 {
		subHeaderLen, subPacket, err = nextSubpacket(contents)
		if err != nil {
			break
		}
		result = append(result, subPacket)
		contents = contents[subHeaderLen+len(subPacket.Contents):]
	}
	return
}

func nextSubpacket(contents []byte) (subHeaderLen int, subPacket *OpaqueSubpacket, err error) {
	// RFC 4880, section 5.2.3.1
	var subLen uint32
	if len(contents) < 1 {
		goto Truncated
	}
	subPacket = &OpaqueSubpacket{}
	switch {
	case contents[0] < 192:
		subHeaderLen = 2 // 1 length byte, 1 subtype byte
		if len(contents) < subHeaderLen {
			goto Truncated
		}
		subLen = uint32(contents[0])
		contents = contents[1:]
	case contents[0] < 255:
		subHeaderLen = 3 // 2 length bytes, 1 subtype
		if len(contents) < subHeaderLen {
			goto Truncated
		}
		subLen = uint32(contents[0]-192)<<8 + uint32(contents[1]) + 192
		contents = contents[2:]
	default:
		subHeaderLen = 6 // 5 length bytes, 1 subtype
		if len(contents) < subHeaderLen {
			goto Truncated
		}
		subLen = uint32(contents[1])<<24 |
			uint32(contents[2])<<16 |
			uint32(contents[3])<<8 |
			uint32(contents[4])
		contents = contents[5:]
	}
	if subLen > uint32(len(contents)) {
		goto Truncated
	}
	subPacket.SubType = contents[0]
	subPacket.Contents = contents[1:subLen]
	return
Truncated:
	err = errors.StructuralError("subpacket truncated")
	return
}

func (osp *OpaqueSubpacket) Serialize(w io.Writer) (err error) {
	buf := make([]byte, 6)
	n := serializeSubpacketLength(buf, len(osp.Contents)+1)
	buf[n] = osp.SubType
	if _, err = w.Write(buf[:n+1]); err != nil {
		return
	}
	_, err = w.Write(osp.Contents)
	return
}
