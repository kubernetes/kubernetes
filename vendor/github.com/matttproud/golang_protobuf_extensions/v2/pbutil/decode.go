// Copyright 2013 Matt T. Proud
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

package pbutil

import (
	"encoding/binary"
	"errors"
	"io"

	"google.golang.org/protobuf/proto"
)

// TODO: Give error package name prefix in next minor release.
var errInvalidVarint = errors.New("invalid varint32 encountered")

// ReadDelimited decodes a message from the provided length-delimited stream,
// where the length is encoded as 32-bit varint prefix to the message body.
// It returns the total number of bytes read and any applicable error.  This is
// roughly equivalent to the companion Java API's
// MessageLite#parseDelimitedFrom.  As per the reader contract, this function
// calls r.Read repeatedly as required until exactly one message including its
// prefix is read and decoded (or an error has occurred).  The function never
// reads more bytes from the stream than required.  The function never returns
// an error if a message has been read and decoded correctly, even if the end
// of the stream has been reached in doing so.  In that case, any subsequent
// calls return (0, io.EOF).
func ReadDelimited(r io.Reader, m proto.Message) (n int, err error) {
	// TODO: Consider allowing the caller to specify a decode buffer in the
	// next major version.

	// TODO: Consider using error wrapping to annotate error state in pass-
	// through cases in the next minor version.

	// Per AbstractParser#parsePartialDelimitedFrom with
	// CodedInputStream#readRawVarint32.
	var headerBuf [binary.MaxVarintLen32]byte
	var bytesRead, varIntBytes int
	var messageLength uint64
	for varIntBytes == 0 { // i.e. no varint has been decoded yet.
		if bytesRead >= len(headerBuf) {
			return bytesRead, errInvalidVarint
		}
		// We have to read byte by byte here to avoid reading more bytes
		// than required. Each read byte is appended to what we have
		// read before.
		newBytesRead, err := r.Read(headerBuf[bytesRead : bytesRead+1])
		if newBytesRead == 0 {
			if err != nil {
				return bytesRead, err
			}
			// A Reader should not return (0, nil); but if it does, it should
			// be treated as no-op according to the Reader contract.
			continue
		}
		bytesRead += newBytesRead
		// Now present everything read so far to the varint decoder and
		// see if a varint can be decoded already.
		messageLength, varIntBytes = binary.Uvarint(headerBuf[:bytesRead])
	}

	messageBuf := make([]byte, messageLength)
	newBytesRead, err := io.ReadFull(r, messageBuf)
	bytesRead += newBytesRead
	if err != nil {
		return bytesRead, err
	}

	return bytesRead, proto.Unmarshal(messageBuf, m)
}
