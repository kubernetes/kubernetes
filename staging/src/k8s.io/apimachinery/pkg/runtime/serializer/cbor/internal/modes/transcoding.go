/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modes

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strconv"
	"sync"
	"unsafe"

	kjson "sigs.k8s.io/json"
)

type TranscodeFunc func(dst io.Writer, src io.Reader) error

func (f TranscodeFunc) Transcode(dst io.Writer, src io.Reader) error {
	return f(dst, src)
}

func TranscodeFromJSON(dst io.Writer, src io.Reader) error {
	inBuf := buffers.Get()
	defer buffers.Put(inBuf)
	if _, err := inBuf.ReadFrom(src); err != nil {
		return err
	}
	data := inBuf.Bytes()
	if ok, err := fastTranscodeFromJSON(data, dst); ok {
		return err
	}

	var tmp any
	dec := kjson.NewDecoderCaseSensitivePreserveInts(bytes.NewReader(data))
	if err := dec.Decode(&tmp); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("extraneous data")
	}

	return encode.MarshalTo(tmp, dst)
}

// AcquireBuffer borrows a pooled internal CBOR buffer that satisfies the
// fast path in EncMode.MarshalTo.
func AcquireBuffer() *bytes.Buffer {
	return &buffers.Get().Buffer
}

// ReleaseBuffer returns a buffer obtained from AcquireBuffer back to the pool.
func ReleaseBuffer(b *bytes.Buffer) {
	b.Reset()
}

// EncodeItemsValueBatched encodes slice elements from itemsValue in batches using
// a single pooled *buffer so EncMode.MarshalTo avoids per-item pool checkout,
// []interface{} slice allocation, and io.Copy overhead while preserving streaming Write calls.
func EncodeItemsValueBatched(w io.Writer, itemsValue reflect.Value, mode EncMode) error {
	n := itemsValue.Len()
	if _, ok := w.(*buffer); ok || n <= 2 {
		for i := 0; i < n; i++ {
			if err := mode.MarshalTo(itemsValue.Index(i).Addr().Interface(), w); err != nil {
				return err
			}
		}
		return nil
	}
	buf := buffers.Get()
	defer buffers.Put(buf)
	const flushThreshold = 32 * 1024
	for i := 0; i < n; i++ {
		if err := mode.delegate.MarshalToBuffer(itemsValue.Index(i).Addr().Interface(), &buf.Buffer); err != nil {
			return err
		}
		if buf.Len() >= flushThreshold {
			if _, err := w.Write(buf.Bytes()); err != nil {
				return err
			}
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		if _, err := w.Write(buf.Bytes()); err != nil {
			return err
		}
		buf.Reset()
	}
	return nil
}

type jsonKV struct {
	key      []byte
	valStart int
	valEnd   int
}

type jsonTranscoder struct {
	data       []byte
	pos        int
	scratch    []byte
	tmpScratch []byte
	strScratch []byte
	kvPool     []jsonKV
}

var transcoderPool = sync.Pool{
	New: func() any {
		return &jsonTranscoder{
			scratch:    make([]byte, 0, 64*1024),
			tmpScratch: make([]byte, 0, 16*1024),
			strScratch: make([]byte, 0, 4*1024),
			kvPool:     make([]jsonKV, 0, 128),
		}
	},
}

func fastTranscodeFromJSON(data []byte, dst io.Writer) (bool, error) {
	if len(data) == 0 {
		return false, nil
	}
	t := transcoderPool.Get().(*jsonTranscoder)
	t.data = data
	t.pos = 0
	t.scratch = t.scratch[:0]
	t.tmpScratch = t.tmpScratch[:0]
	t.strScratch = t.strScratch[:0]
	t.kvPool = t.kvPool[:0]
	defer func() {
		t.data = nil
		if cap(t.scratch) <= 1024*1024 && cap(t.tmpScratch) <= 512*1024 && cap(t.strScratch) <= 64*1024 && cap(t.kvPool) <= 2048 {
			transcoderPool.Put(t)
		}
	}()

	if !t.transcodeValue() {
		return false, nil
	}
	t.skipWS()
	if t.pos != len(t.data) {
		return false, nil
	}
	_, err := dst.Write(t.scratch)
	return true, err
}

func (t *jsonTranscoder) skipWS() {
	for t.pos < len(t.data) {
		c := t.data[t.pos]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		t.pos++
	}
}

func (t *jsonTranscoder) transcodeValue() bool {
	t.skipWS()
	if t.pos >= len(t.data) {
		return false
	}
	switch t.data[t.pos] {
	case '{':
		return t.transcodeObject()
	case '[':
		return t.transcodeArray()
	case '"':
		s, ok := t.parseSimpleString()
		if !ok {
			return false
		}
		appendCBORHead(&t.scratch, 0x40, uint64(len(s)))
		t.scratch = append(t.scratch, s...)
		return true
	case 't':
		if t.pos+4 <= len(t.data) && string(t.data[t.pos:t.pos+4]) == "true" {
			t.pos += 4
			t.scratch = append(t.scratch, 0xf5)
			return true
		}
		return false
	case 'f':
		if t.pos+5 <= len(t.data) && string(t.data[t.pos:t.pos+5]) == "false" {
			t.pos += 5
			t.scratch = append(t.scratch, 0xf4)
			return true
		}
		return false
	case 'n':
		if t.pos+4 <= len(t.data) && string(t.data[t.pos:t.pos+4]) == "null" {
			t.pos += 4
			t.scratch = append(t.scratch, 0xf6)
			return true
		}
		return false
	default:
		c := t.data[t.pos]
		if c == '-' || (c >= '0' && c <= '9') {
			return t.transcodeNumber()
		}
		return false
	}
}

func (t *jsonTranscoder) transcodeObject() bool {
	t.pos++ // skip '{'
	t.skipWS()
	if t.pos >= len(t.data) {
		return false
	}
	if t.data[t.pos] == '}' {
		t.pos++
		t.scratch = append(t.scratch, 0xa0)
		return true
	}

	scratchBase := len(t.scratch)
	kvBase := len(t.kvPool)
	for {
		t.skipWS()
		if t.pos >= len(t.data) || t.data[t.pos] != '"' {
			return false
		}
		key, ok := t.parseSimpleString()
		if !ok {
			return false
		}
		t.skipWS()
		if t.pos >= len(t.data) || t.data[t.pos] != ':' {
			return false
		}
		t.pos++ // skip ':'

		valStart := len(t.scratch)
		if !t.transcodeValue() {
			return false
		}
		valEnd := len(t.scratch)

		dupIdx := -1
		for i := kvBase; i < len(t.kvPool); i++ {
			if bytes.Equal(t.kvPool[i].key, key) {
				dupIdx = i
				break
			}
		}
		if dupIdx >= 0 {
			t.kvPool[dupIdx].valStart = valStart
			t.kvPool[dupIdx].valEnd = valEnd
		} else {
			t.kvPool = append(t.kvPool, jsonKV{key: key, valStart: valStart, valEnd: valEnd})
		}

		t.skipWS()
		if t.pos >= len(t.data) {
			return false
		}
		if t.data[t.pos] == ',' {
			t.pos++
			continue
		}
		if t.data[t.pos] == '}' {
			t.pos++
			break
		}
		return false
	}

	kvs := t.kvPool[kvBase:]
	sortJSONKVs(kvs)

	t.tmpScratch = append(t.tmpScratch[:0], t.scratch[scratchBase:]...)
	t.scratch = t.scratch[:scratchBase]
	appendCBORHead(&t.scratch, 0xa0, uint64(len(kvs)))
	for i := range kvs {
		kv := kvs[i]
		appendCBORHead(&t.scratch, 0x40, uint64(len(kv.key)))
		t.scratch = append(t.scratch, kv.key...)
		t.scratch = append(t.scratch, t.tmpScratch[kv.valStart-scratchBase:kv.valEnd-scratchBase]...)
	}
	t.kvPool = t.kvPool[:kvBase]
	return true
}

func sortJSONKVs(kvs []jsonKV) {
	for i := 1; i < len(kvs); i++ {
		cur := kvs[i]
		j := i - 1
		for j >= 0 && jsonKVLess(cur.key, kvs[j].key) {
			kvs[j+1] = kvs[j]
			j--
		}
		kvs[j+1] = cur
	}
}

func jsonKVLess(a, b []byte) bool {
	if len(a) != len(b) {
		return len(a) < len(b)
	}
	return bytes.Compare(a, b) < 0
}

func (t *jsonTranscoder) transcodeArray() bool {
	t.pos++ // skip '['
	t.skipWS()
	if t.pos >= len(t.data) {
		return false
	}
	if t.data[t.pos] == ']' {
		t.pos++
		t.scratch = append(t.scratch, 0x80)
		return true
	}

	scratchBase := len(t.scratch)
	t.scratch = append(t.scratch, 0) // reserve 1 byte for header
	count := uint64(0)
	for {
		if !t.transcodeValue() {
			return false
		}
		count++
		t.skipWS()
		if t.pos >= len(t.data) {
			return false
		}
		if t.data[t.pos] == ',' {
			t.pos++
			continue
		}
		if t.data[t.pos] == ']' {
			t.pos++
			break
		}
		return false
	}

	if count <= 23 {
		t.scratch[scratchBase] = 0x80 + byte(count)
	} else {
		var hdr [9]byte
		hdrLen := putCBORHead(hdr[:], 0x80, count)
		extra := hdrLen - 1
		origLen := len(t.scratch)
		t.scratch = append(t.scratch, hdr[:extra]...)
		copy(t.scratch[scratchBase+hdrLen:], t.scratch[scratchBase+1:origLen])
		copy(t.scratch[scratchBase:scratchBase+hdrLen], hdr[:hdrLen])
	}
	return true
}

func (t *jsonTranscoder) parseSimpleString() ([]byte, bool) {
	t.pos++ // skip opening '"'
	start := t.pos
	for t.pos < len(t.data) {
		c := t.data[t.pos]
		if c == '"' {
			s := t.data[start:t.pos]
			t.pos++
			return s, true
		}
		if c == '\\' {
			return t.parseEscapedString(start)
		}
		if c < 0x20 || c >= 0x80 {
			return nil, false
		}
		t.pos++
	}
	return nil, false
}

func (t *jsonTranscoder) parseEscapedString(start int) ([]byte, bool) {
	strBase := len(t.strScratch)
	t.strScratch = append(t.strScratch, t.data[start:t.pos]...)
	for t.pos < len(t.data) {
		c := t.data[t.pos]
		if c == '"' {
			t.pos++
			return t.strScratch[strBase:], true
		}
		if c == '\\' {
			t.pos++
			if t.pos >= len(t.data) {
				return nil, false
			}
			esc := t.data[t.pos]
			switch esc {
			case '"', '\\', '/':
				t.strScratch = append(t.strScratch, esc)
			case 'b':
				t.strScratch = append(t.strScratch, '\b')
			case 'f':
				t.strScratch = append(t.strScratch, '\f')
			case 'n':
				t.strScratch = append(t.strScratch, '\n')
			case 'r':
				t.strScratch = append(t.strScratch, '\r')
			case 't':
				t.strScratch = append(t.strScratch, '\t')
			default:
				return nil, false
			}
			t.pos++
			continue
		}
		if c < 0x20 || c >= 0x80 {
			return nil, false
		}
		t.strScratch = append(t.strScratch, c)
		t.pos++
	}
	return nil, false
}

func (t *jsonTranscoder) transcodeNumber() bool {
	start := t.pos
	if t.data[t.pos] == '-' {
		t.pos++
		if t.pos >= len(t.data) {
			return false
		}
	}
	if t.data[t.pos] == '0' {
		t.pos++
		if t.pos < len(t.data) && t.data[t.pos] >= '0' && t.data[t.pos] <= '9' {
			return false
		}
	} else if t.data[t.pos] >= '1' && t.data[t.pos] <= '9' {
		for t.pos < len(t.data) && t.data[t.pos] >= '0' && t.data[t.pos] <= '9' {
			t.pos++
		}
	} else {
		return false
	}

	hasDot := false
	hasExp := false
	if t.pos < len(t.data) && t.data[t.pos] == '.' {
		hasDot = true
		t.pos++
		if t.pos >= len(t.data) || t.data[t.pos] < '0' || t.data[t.pos] > '9' {
			return false
		}
		for t.pos < len(t.data) && t.data[t.pos] >= '0' && t.data[t.pos] <= '9' {
			t.pos++
		}
	}
	if t.pos < len(t.data) && (t.data[t.pos] == 'e' || t.data[t.pos] == 'E') {
		hasExp = true
		t.pos++
		if t.pos < len(t.data) && (t.data[t.pos] == '+' || t.data[t.pos] == '-') {
			t.pos++
		}
		if t.pos >= len(t.data) || t.data[t.pos] < '0' || t.data[t.pos] > '9' {
			return false
		}
		for t.pos < len(t.data) && t.data[t.pos] >= '0' && t.data[t.pos] <= '9' {
			t.pos++
		}
	}

	numStr := string(t.data[start:t.pos])
	if !hasDot && !hasExp {
		if iv, err := strconv.ParseInt(numStr, 10, 64); err == nil {
			if iv >= 0 {
				appendCBORHead(&t.scratch, 0x00, uint64(iv))
			} else {
				appendCBORHead(&t.scratch, 0x20, uint64(-1-iv))
			}
			return true
		}
	}
	fv, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return false
	}
	buf := bytes.NewBuffer(t.scratch)
	if err := encode.delegate.MarshalToBuffer(fv, buf); err != nil {
		return false
	}
	t.scratch = buf.Bytes()
	return true
}

func appendCBORHead(dst *[]byte, base byte, n uint64) {
	var hdr [9]byte
	hdrLen := putCBORHead(hdr[:], base, n)
	*dst = append(*dst, hdr[:hdrLen]...)
}

func putCBORHead(dst []byte, base byte, n uint64) int {
	switch {
	case n <= 23:
		dst[0] = base + byte(n)
		return 1
	case n <= 0xFF:
		dst[0] = base | 24
		dst[1] = byte(n)
		return 2
	case n <= 0xFFFF:
		dst[0] = base | 25
		dst[1] = byte(n >> 8)
		dst[2] = byte(n)
		return 3
	case n <= 0xFFFFFFFF:
		dst[0] = base | 26
		dst[1] = byte(n >> 24)
		dst[2] = byte(n >> 16)
		dst[3] = byte(n >> 8)
		dst[4] = byte(n)
		return 5
	default:
		dst[0] = base | 27
		dst[1] = byte(n >> 56)
		dst[2] = byte(n >> 48)
		dst[3] = byte(n >> 40)
		dst[4] = byte(n >> 32)
		dst[5] = byte(n >> 24)
		dst[6] = byte(n >> 16)
		dst[7] = byte(n >> 8)
		dst[8] = byte(n)
		return 9
	}
}

func TranscodeToJSON(dst io.Writer, src io.Reader) error {
	var tmp any
	dec := decode.NewDecoder(src)
	if err := dec.Decode(&tmp); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("extraneous data")
	}

	// Use an Encoder to avoid the extra []byte allocated by Marshal. Encode, unlike Marshal,
	// appends a trailing newline to separate consecutive encodings of JSON values that aren't
	// self-delimiting, like numbers. Strip the newline to avoid the assumption that every
	// json.Unmarshaler implementation will accept trailing whitespace.
	enc := json.NewEncoder(&trailingLinefeedSuppressor{delegate: dst})
	enc.SetIndent("", "")
	return enc.Encode(tmp)
}

// trailingLinefeedSuppressor is an io.Writer that wraps another io.Writer, suppressing a single
// trailing linefeed if it is the last byte written by the latest call to Write.
type trailingLinefeedSuppressor struct {
	lf       bool
	delegate io.Writer
}

func (w *trailingLinefeedSuppressor) Write(p []byte) (int, error) {
	if len(p) == 0 {
		// Avoid flushing a buffered linefeeds on an empty write.
		return 0, nil
	}

	if w.lf {
		// The previous write had a trailing linefeed that was buffered. That wasn't the
		// last Write call, so flush the buffered linefeed before continuing.
		n, err := w.delegate.Write([]byte{'\n'})
		if n > 0 {
			w.lf = false
		}
		if err != nil {
			return 0, err
		}
	}

	if p[len(p)-1] != '\n' {
		return w.delegate.Write(p)
	}

	p = p[:len(p)-1]

	if len(p) == 0 { // []byte{'\n'}
		w.lf = true
		return 1, nil
	}

	n, err := w.delegate.Write(p)
	if n == len(p) {
		// Everything up to the trailing linefeed has been flushed. Eat the linefeed.
		w.lf = true
		n++
	}
	return n, err
}

// TranscodeJSONObject transcodes a raw JSON object string directly into CBOR bytes
// without unmarshaling into map[string]interface{}. Returns (nil, false) if rawStr
// is not a valid JSON object or null so the caller can fall back to utiljson.Unmarshal.
func TranscodeJSONObject(rawStr string) ([]byte, bool) {
	if len(rawStr) == 0 {
		return nil, false
	}
	t := transcoderPool.Get().(*jsonTranscoder)
	t.data = unsafe.Slice(unsafe.StringData(rawStr), len(rawStr))
	t.pos = 0
	t.scratch = t.scratch[:0]
	t.tmpScratch = t.tmpScratch[:0]
	t.strScratch = t.strScratch[:0]
	t.kvPool = t.kvPool[:0]
	t.skipWS()
	if t.pos >= len(t.data) || (t.data[t.pos] != '{' && t.data[t.pos] != 'n') {
		t.data = nil
		transcoderPool.Put(t)
		return nil, false
	}
	ok := t.transcodeValue()
	if ok {
		t.skipWS()
		ok = t.pos == len(t.data)
	}
	var out []byte
	if ok {
		out = bytes.Clone(t.scratch)
	}
	t.data = nil
	if cap(t.scratch) <= 1024*1024 && cap(t.tmpScratch) <= 512*1024 && cap(t.strScratch) <= 64*1024 && cap(t.kvPool) <= 2048 {
		transcoderPool.Put(t)
	}
	return out, ok
}
