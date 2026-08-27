/*-
 * Copyright 2014 Square Inc.
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
 */

package jose

import (
	"bytes"
	"compress/flate"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"math/big"
	"strings"
	"unicode"

	"github.com/go-jose/go-jose/v4/json"
)

// Helper function to serialize known-good objects.
// Precondition: value is not a nil pointer.
func mustSerializeJSON(value interface{}) []byte {
	out, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	// We never want to serialize the top-level value "null," since it's not a
	// valid JOSE message. But if a caller passes in a nil pointer to this method,
	// MarshalJSON will happily serialize it as the top-level value "null". If
	// that value is then embedded in another operation, for instance by being
	// base64-encoded and fed as input to a signing algorithm
	// (https://github.com/go-jose/go-jose/issues/22), the result will be
	// incorrect. Because this method is intended for known-good objects, and a nil
	// pointer is not a known-good object, we are free to panic in this case.
	// Note: It's not possible to directly check whether the data pointed at by an
	// interface is a nil pointer, so we do this hacky workaround.
	// https://groups.google.com/forum/#!topic/golang-nuts/wnH302gBa4I
	if string(out) == "null" {
		panic("Tried to serialize a nil pointer.")
	}
	return out
}

// Strip all newlines and whitespace
func stripWhitespace(data string) string {
	buf := strings.Builder{}
	buf.Grow(len(data))
	for _, r := range data {
		if !unicode.IsSpace(r) {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}

// Perform compression based on algorithm
func compress(algorithm CompressionAlgorithm, input []byte) ([]byte, error) {
	switch algorithm {
	case DEFLATE:
		return deflate(input)
	default:
		return nil, ErrUnsupportedAlgorithm
	}
}

// Perform decompression based on algorithm
func decompress(algorithm CompressionAlgorithm, input []byte) ([]byte, error) {
	switch algorithm {
	case DEFLATE:
		return inflate(input)
	default:
		return nil, ErrUnsupportedAlgorithm
	}
}

// deflate compresses the input.
func deflate(input []byte) ([]byte, error) {
	output := new(bytes.Buffer)

	// Writing to byte buffer, err is always nil
	writer, _ := flate.NewWriter(output, 1)
	_, _ = io.Copy(writer, bytes.NewBuffer(input))

	err := writer.Close()
	return output.Bytes(), err
}

// inflate decompresses the input.
//
// Errors if the decompressed data would be >250kB or >10x the size of the
// compressed data, whichever is larger.
func inflate(input []byte) ([]byte, error) {
	output := new(bytes.Buffer)
	reader := flate.NewReader(bytes.NewBuffer(input))

	maxCompressedSize := max(250_000, 10*int64(len(input)))

	limit := maxCompressedSize + 1
	n, err := io.CopyN(output, reader, limit)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if n == limit {
		return nil, fmt.Errorf("uncompressed data would be too large (>%d bytes)", maxCompressedSize)
	}

	err = reader.Close()
	return output.Bytes(), err
}

// byteBuffer represents a slice of bytes that can be serialized to url-safe base64.
type byteBuffer struct {
	data []byte
}

func newBuffer(data []byte) *byteBuffer {
	if data == nil {
		return nil
	}
	return &byteBuffer{
		data: data,
	}
}

func newFixedSizeBuffer(data []byte, length int) *byteBuffer {
	if len(data) > length {
		panic("go-jose/go-jose: invalid call to newFixedSizeBuffer (len(data) > length)")
	}
	pad := make([]byte, length-len(data))
	return newBuffer(append(pad, data...))
}

func newBufferFromInt(num uint64) *byteBuffer {
	data := make([]byte, 8)
	binary.BigEndian.PutUint64(data, num)
	return newBuffer(bytes.TrimLeft(data, "\x00"))
}

func (b *byteBuffer) MarshalJSON() ([]byte, error) {
	return json.Marshal(b.base64())
}

func (b *byteBuffer) UnmarshalJSON(data []byte) error {
	var encoded string
	err := json.Unmarshal(data, &encoded)
	if err != nil {
		return err
	}

	if encoded == "" {
		return nil
	}

	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return err
	}

	*b = *newBuffer(decoded)

	return nil
}

func (b *byteBuffer) base64() string {
	return base64.RawURLEncoding.EncodeToString(b.data)
}

func (b *byteBuffer) bytes() []byte {
	// Handling nil here allows us to transparently handle nil slices when serializing.
	if b == nil {
		return nil
	}
	return b.data
}

func (b byteBuffer) bigInt() *big.Int {
	return new(big.Int).SetBytes(b.data)
}

func (b byteBuffer) toInt() int {
	return int(b.bigInt().Int64())
}

func base64EncodeLen(sl []byte) int {
	return base64.RawURLEncoding.EncodedLen(len(sl))
}

func base64JoinWithDots(inputs ...[]byte) string {
	if len(inputs) == 0 {
		return ""
	}

	// Count of dots.
	totalCount := len(inputs) - 1

	for _, input := range inputs {
		totalCount += base64EncodeLen(input)
	}

	out := make([]byte, totalCount)
	startEncode := 0
	for i, input := range inputs {
		base64.RawURLEncoding.Encode(out[startEncode:], input)

		if i == len(inputs)-1 {
			continue
		}

		startEncode += base64EncodeLen(input)
		out[startEncode] = '.'
		startEncode++
	}

	return string(out)
}
