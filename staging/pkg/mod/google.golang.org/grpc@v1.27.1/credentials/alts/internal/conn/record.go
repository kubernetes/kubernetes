/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package conn contains an implementation of a secure channel created by gRPC
// handshakers.
package conn

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"

	core "google.golang.org/grpc/credentials/alts/internal"
)

// ALTSRecordCrypto is the interface for gRPC ALTS record protocol.
type ALTSRecordCrypto interface {
	// Encrypt encrypts the plaintext and computes the tag (if any) of dst
	// and plaintext, dst and plaintext do not overlap.
	Encrypt(dst, plaintext []byte) ([]byte, error)
	// EncryptionOverhead returns the tag size (if any) in bytes.
	EncryptionOverhead() int
	// Decrypt decrypts ciphertext and verify the tag (if any). dst and
	// ciphertext may alias exactly or not at all. To reuse ciphertext's
	// storage for the decrypted output, use ciphertext[:0] as dst.
	Decrypt(dst, ciphertext []byte) ([]byte, error)
}

// ALTSRecordFunc is a function type for factory functions that create
// ALTSRecordCrypto instances.
type ALTSRecordFunc func(s core.Side, keyData []byte) (ALTSRecordCrypto, error)

const (
	// MsgLenFieldSize is the byte size of the frame length field of a
	// framed message.
	MsgLenFieldSize = 4
	// The byte size of the message type field of a framed message.
	msgTypeFieldSize = 4
	// The bytes size limit for a ALTS record message.
	altsRecordLengthLimit = 1024 * 1024 // 1 MiB
	// The default bytes size of a ALTS record message.
	altsRecordDefaultLength = 4 * 1024 // 4KiB
	// Message type value included in ALTS record framing.
	altsRecordMsgType = uint32(0x06)
	// The initial write buffer size.
	altsWriteBufferInitialSize = 32 * 1024 // 32KiB
	// The maximum write buffer size. This *must* be multiple of
	// altsRecordDefaultLength.
	altsWriteBufferMaxSize = 512 * 1024 // 512KiB
)

var (
	protocols = make(map[string]ALTSRecordFunc)
)

// RegisterProtocol register a ALTS record encryption protocol.
func RegisterProtocol(protocol string, f ALTSRecordFunc) error {
	if _, ok := protocols[protocol]; ok {
		return fmt.Errorf("protocol %v is already registered", protocol)
	}
	protocols[protocol] = f
	return nil
}

// conn represents a secured connection. It implements the net.Conn interface.
type conn struct {
	net.Conn
	crypto ALTSRecordCrypto
	// buf holds data that has been read from the connection and decrypted,
	// but has not yet been returned by Read.
	buf                []byte
	payloadLengthLimit int
	// protected holds data read from the network but have not yet been
	// decrypted. This data might not compose a complete frame.
	protected []byte
	// writeBuf is a buffer used to contain encrypted frames before being
	// written to the network.
	writeBuf []byte
	// nextFrame stores the next frame (in protected buffer) info.
	nextFrame []byte
	// overhead is the calculated overhead of each frame.
	overhead int
}

// NewConn creates a new secure channel instance given the other party role and
// handshaking result.
func NewConn(c net.Conn, side core.Side, recordProtocol string, key []byte, protected []byte) (net.Conn, error) {
	newCrypto := protocols[recordProtocol]
	if newCrypto == nil {
		return nil, fmt.Errorf("negotiated unknown next_protocol %q", recordProtocol)
	}
	crypto, err := newCrypto(side, key)
	if err != nil {
		return nil, fmt.Errorf("protocol %q: %v", recordProtocol, err)
	}
	overhead := MsgLenFieldSize + msgTypeFieldSize + crypto.EncryptionOverhead()
	payloadLengthLimit := altsRecordDefaultLength - overhead
	if protected == nil {
		// We pre-allocate protected to be of size
		// 2*altsRecordDefaultLength-1 during initialization. We only
		// read from the network into protected when protected does not
		// contain a complete frame, which is at most
		// altsRecordDefaultLength-1 (bytes). And we read at most
		// altsRecordDefaultLength (bytes) data into protected at one
		// time. Therefore, 2*altsRecordDefaultLength-1 is large enough
		// to buffer data read from the network.
		protected = make([]byte, 0, 2*altsRecordDefaultLength-1)
	}

	altsConn := &conn{
		Conn:               c,
		crypto:             crypto,
		payloadLengthLimit: payloadLengthLimit,
		protected:          protected,
		writeBuf:           make([]byte, altsWriteBufferInitialSize),
		nextFrame:          protected,
		overhead:           overhead,
	}
	return altsConn, nil
}

// Read reads and decrypts a frame from the underlying connection, and copies the
// decrypted payload into b. If the size of the payload is greater than len(b),
// Read retains the remaining bytes in an internal buffer, and subsequent calls
// to Read will read from this buffer until it is exhausted.
func (p *conn) Read(b []byte) (n int, err error) {
	if len(p.buf) == 0 {
		var framedMsg []byte
		framedMsg, p.nextFrame, err = ParseFramedMsg(p.nextFrame, altsRecordLengthLimit)
		if err != nil {
			return n, err
		}
		// Check whether the next frame to be decrypted has been
		// completely received yet.
		if len(framedMsg) == 0 {
			copy(p.protected, p.nextFrame)
			p.protected = p.protected[:len(p.nextFrame)]
			// Always copy next incomplete frame to the beginning of
			// the protected buffer and reset nextFrame to it.
			p.nextFrame = p.protected
		}
		// Check whether a complete frame has been received yet.
		for len(framedMsg) == 0 {
			if len(p.protected) == cap(p.protected) {
				tmp := make([]byte, len(p.protected), cap(p.protected)+altsRecordDefaultLength)
				copy(tmp, p.protected)
				p.protected = tmp
			}
			n, err = p.Conn.Read(p.protected[len(p.protected):min(cap(p.protected), len(p.protected)+altsRecordDefaultLength)])
			if err != nil {
				return 0, err
			}
			p.protected = p.protected[:len(p.protected)+n]
			framedMsg, p.nextFrame, err = ParseFramedMsg(p.protected, altsRecordLengthLimit)
			if err != nil {
				return 0, err
			}
		}
		// Now we have a complete frame, decrypted it.
		msg := framedMsg[MsgLenFieldSize:]
		msgType := binary.LittleEndian.Uint32(msg[:msgTypeFieldSize])
		if msgType&0xff != altsRecordMsgType {
			return 0, fmt.Errorf("received frame with incorrect message type %v, expected lower byte %v",
				msgType, altsRecordMsgType)
		}
		ciphertext := msg[msgTypeFieldSize:]

		// Decrypt requires that if the dst and ciphertext alias, they
		// must alias exactly. Code here used to use msg[:0], but msg
		// starts MsgLenFieldSize+msgTypeFieldSize bytes earlier than
		// ciphertext, so they alias inexactly. Using ciphertext[:0]
		// arranges the appropriate aliasing without needing to copy
		// ciphertext or use a separate destination buffer. For more info
		// check: https://golang.org/pkg/crypto/cipher/#AEAD.
		p.buf, err = p.crypto.Decrypt(ciphertext[:0], ciphertext)
		if err != nil {
			return 0, err
		}
	}

	n = copy(b, p.buf)
	p.buf = p.buf[n:]
	return n, nil
}

// Write encrypts, frames, and writes bytes from b to the underlying connection.
func (p *conn) Write(b []byte) (n int, err error) {
	n = len(b)
	// Calculate the output buffer size with framing and encryption overhead.
	numOfFrames := int(math.Ceil(float64(len(b)) / float64(p.payloadLengthLimit)))
	size := len(b) + numOfFrames*p.overhead
	// If writeBuf is too small, increase its size up to the maximum size.
	partialBSize := len(b)
	if size > altsWriteBufferMaxSize {
		size = altsWriteBufferMaxSize
		const numOfFramesInMaxWriteBuf = altsWriteBufferMaxSize / altsRecordDefaultLength
		partialBSize = numOfFramesInMaxWriteBuf * p.payloadLengthLimit
	}
	if len(p.writeBuf) < size {
		p.writeBuf = make([]byte, size)
	}

	for partialBStart := 0; partialBStart < len(b); partialBStart += partialBSize {
		partialBEnd := partialBStart + partialBSize
		if partialBEnd > len(b) {
			partialBEnd = len(b)
		}
		partialB := b[partialBStart:partialBEnd]
		writeBufIndex := 0
		for len(partialB) > 0 {
			payloadLen := len(partialB)
			if payloadLen > p.payloadLengthLimit {
				payloadLen = p.payloadLengthLimit
			}
			buf := partialB[:payloadLen]
			partialB = partialB[payloadLen:]

			// Write buffer contains: length, type, payload, and tag
			// if any.

			// 1. Fill in type field.
			msg := p.writeBuf[writeBufIndex+MsgLenFieldSize:]
			binary.LittleEndian.PutUint32(msg, altsRecordMsgType)

			// 2. Encrypt the payload and create a tag if any.
			msg, err = p.crypto.Encrypt(msg[:msgTypeFieldSize], buf)
			if err != nil {
				return n, err
			}

			// 3. Fill in the size field.
			binary.LittleEndian.PutUint32(p.writeBuf[writeBufIndex:], uint32(len(msg)))

			// 4. Increase writeBufIndex.
			writeBufIndex += len(buf) + p.overhead
		}
		nn, err := p.Conn.Write(p.writeBuf[:writeBufIndex])
		if err != nil {
			// We need to calculate the actual data size that was
			// written. This means we need to remove header,
			// encryption overheads, and any partially-written
			// frame data.
			numOfWrittenFrames := int(math.Floor(float64(nn) / float64(altsRecordDefaultLength)))
			return partialBStart + numOfWrittenFrames*p.payloadLengthLimit, err
		}
	}
	return n, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
