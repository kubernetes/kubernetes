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
	"strings"
	"testing"
)

func TestDeflateRoundtrip(t *testing.T) {
	original := []byte("Lorem ipsum dolor sit amet")

	compressed, err := deflate(original)
	if err != nil {
		panic(err)
	}

	output, err := inflate(compressed)
	if err != nil {
		panic(err)
	}

	if bytes.Compare(output, original) != 0 {
		t.Error("Input and output do not match")
	}
}

func TestInvalidCompression(t *testing.T) {
	_, err := compress("XYZ", []byte{})
	if err == nil {
		t.Error("should not accept invalid algorithm")
	}

	_, err = decompress("XYZ", []byte{})
	if err == nil {
		t.Error("should not accept invalid algorithm")
	}

	_, err = decompress(DEFLATE, []byte{1, 2, 3, 4})
	if err == nil {
		t.Error("should not accept invalid data")
	}
}

func TestByteBufferTrim(t *testing.T) {
	buf := newBufferFromInt(1)
	if !bytes.Equal(buf.data, []byte{1}) {
		t.Error("Byte buffer for integer '1' should contain [0x01]")
	}

	buf = newBufferFromInt(65537)
	if !bytes.Equal(buf.data, []byte{1, 0, 1}) {
		t.Error("Byte buffer for integer '65537' should contain [0x01, 0x00, 0x01]")
	}
}

func TestFixedSizeBuffer(t *testing.T) {
	data0 := []byte{}
	data1 := []byte{1}
	data2 := []byte{1, 2}
	data3 := []byte{1, 2, 3}
	data4 := []byte{1, 2, 3, 4}

	buf0 := newFixedSizeBuffer(data0, 4)
	buf1 := newFixedSizeBuffer(data1, 4)
	buf2 := newFixedSizeBuffer(data2, 4)
	buf3 := newFixedSizeBuffer(data3, 4)
	buf4 := newFixedSizeBuffer(data4, 4)

	if !bytes.Equal(buf0.data, []byte{0, 0, 0, 0}) {
		t.Error("Invalid padded buffer for buf0")
	}
	if !bytes.Equal(buf1.data, []byte{0, 0, 0, 1}) {
		t.Error("Invalid padded buffer for buf1")
	}
	if !bytes.Equal(buf2.data, []byte{0, 0, 1, 2}) {
		t.Error("Invalid padded buffer for buf2")
	}
	if !bytes.Equal(buf3.data, []byte{0, 1, 2, 3}) {
		t.Error("Invalid padded buffer for buf3")
	}
	if !bytes.Equal(buf4.data, []byte{1, 2, 3, 4}) {
		t.Error("Invalid padded buffer for buf4")
	}
}

func TestSerializeJSONRejectsNil(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil || !strings.Contains(r.(string), "nil pointer") {
			t.Error("serialize function should not accept nil pointer")
		}
	}()

	mustSerializeJSON(nil)
}

func TestFixedSizeBufferTooLarge(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("should not be able to create fixed size buffer with oversized data")
		}
	}()

	newFixedSizeBuffer(make([]byte, 2), 1)
}
