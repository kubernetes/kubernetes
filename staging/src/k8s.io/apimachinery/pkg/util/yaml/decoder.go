/*
Copyright 2014 The Kubernetes Authors.

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

package yaml

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"

	jsonutil "k8s.io/apimachinery/pkg/util/json"

	"sigs.k8s.io/yaml"
)

// Unmarshal unmarshals the given data
// If v is a *map[string]interface{}, *[]interface{}, or *interface{} numbers
// are converted to int64 or float64
func Unmarshal(data []byte, v interface{}) error {
	preserveIntFloat := func(d *json.Decoder) *json.Decoder {
		d.UseNumber()
		return d
	}
	switch v := v.(type) {
	case *map[string]interface{}:
		if err := yaml.Unmarshal(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertMapNumbers(*v, 0)
	case *[]interface{}:
		if err := yaml.Unmarshal(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertSliceNumbers(*v, 0)
	case *interface{}:
		if err := yaml.Unmarshal(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertInterfaceNumbers(v, 0)
	default:
		return yaml.Unmarshal(data, v)
	}
}

// UnmarshalStrict unmarshals the given data
// strictly (erroring when there are duplicate fields).
func UnmarshalStrict(data []byte, v interface{}) error {
	preserveIntFloat := func(d *json.Decoder) *json.Decoder {
		d.UseNumber()
		return d
	}
	switch v := v.(type) {
	case *map[string]interface{}:
		if err := yaml.UnmarshalStrict(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertMapNumbers(*v, 0)
	case *[]interface{}:
		if err := yaml.UnmarshalStrict(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertSliceNumbers(*v, 0)
	case *interface{}:
		if err := yaml.UnmarshalStrict(data, v, preserveIntFloat); err != nil {
			return err
		}
		return jsonutil.ConvertInterfaceNumbers(v, 0)
	default:
		return yaml.UnmarshalStrict(data, v)
	}
}

// ToJSON converts a single YAML document into a JSON document
// or returns an error. If the document appears to be JSON the
// YAML decoding path is not used (so that error messages are
// JSON specific).
func ToJSON(data []byte) ([]byte, error) {
	if IsJSONBuffer(data) {
		return data, nil
	}
	return yaml.YAMLToJSON(data)
}

// YAMLToJSONDecoder decodes YAML documents from an io.Reader by
// separating individual documents. It first converts the YAML
// body to JSON, then unmarshals the JSON.
type YAMLToJSONDecoder struct {
	reader      Reader
	inputOffset int
}

// NewYAMLToJSONDecoder decodes YAML documents from the provided
// stream in chunks by converting each document (as defined by
// the YAML spec) into its own chunk, converting it to JSON via
// yaml.YAMLToJSON, and then passing it to json.Decoder.
func NewYAMLToJSONDecoder(r io.Reader) *YAMLToJSONDecoder {
	reader := bufio.NewReader(r)
	return &YAMLToJSONDecoder{
		reader: NewYAMLReader(reader),
	}
}

// Decode reads a YAML document as JSON from the stream or returns
// an error. The decoding rules match json.Unmarshal, not
// yaml.Unmarshal.
func (d *YAMLToJSONDecoder) Decode(into interface{}) error {
	bytes, err := d.reader.Read()
	if err != nil && err != io.EOF { //nolint:errorlint
		return err
	}

	if len(bytes) != 0 {
		err := yaml.Unmarshal(bytes, into)
		if err != nil {
			return YAMLSyntaxError{err}
		}
	}
	d.inputOffset += len(bytes)
	return err
}

func (d *YAMLToJSONDecoder) InputOffset() int {
	return d.inputOffset
}

// YAMLDecoder reads chunks of objects and returns ErrShortBuffer if
// the data is not sufficient.
type YAMLDecoder struct {
	r         io.ReadCloser
	scanner   *bufio.Scanner
	remaining []byte
}

// NewDocumentDecoder decodes YAML documents from the provided
// stream in chunks by converting each document (as defined by
// the YAML spec) into its own chunk. io.ErrShortBuffer will be
// returned if the entire buffer could not be read to assist
// the caller in framing the chunk.
func NewDocumentDecoder(r io.ReadCloser) io.ReadCloser {
	scanner := bufio.NewScanner(r)
	// the size of initial allocation for buffer 4k
	buf := make([]byte, 4*1024)
	// the maximum size used to buffer a token 5M
	scanner.Buffer(buf, 5*1024*1024)
	scanner.Split(splitYAMLDocument)
	return &YAMLDecoder{
		r:       r,
		scanner: scanner,
	}
}

// Read reads the previous slice into the buffer, or attempts to read
// the next chunk.
// TODO: switch to readline approach.
func (d *YAMLDecoder) Read(data []byte) (n int, err error) {
	left := len(d.remaining)
	if left == 0 {
		// return the next chunk from the stream
		if !d.scanner.Scan() {
			err := d.scanner.Err()
			if err == nil {
				err = io.EOF
			}
			return 0, err
		}
		out := d.scanner.Bytes()
		d.remaining = out
		left = len(out)
	}

	// fits within data
	if left <= len(data) {
		copy(data, d.remaining)
		d.remaining = nil
		return left, nil
	}

	// caller will need to reread
	copy(data, d.remaining[:len(data)])
	d.remaining = d.remaining[len(data):]
	return len(data), io.ErrShortBuffer
}

func (d *YAMLDecoder) Close() error {
	return d.r.Close()
}

const yamlSeparator = "\n---"
const separator = "---"

// splitYAMLDocument is a bufio.SplitFunc for splitting YAML streams into individual documents.
func splitYAMLDocument(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	sep := len([]byte(yamlSeparator))
	if i := bytes.Index(data, []byte(yamlSeparator)); i >= 0 {
		// We have a potential document terminator
		i += sep
		after := data[i:]
		if len(after) == 0 {
			// we can't read any more characters
			if atEOF {
				return len(data), data[:len(data)-sep], nil
			}
			return 0, nil, nil
		}
		if j := bytes.IndexByte(after, '\n'); j >= 0 {
			return i + j + 1, data[0 : i-sep], nil
		}
		return 0, nil, nil
	}
	// If we're at EOF, we have a final, non-terminated line. Return it.
	if atEOF {
		return len(data), data, nil
	}
	// Request more data.
	return 0, nil, nil
}

// YAMLOrJSONDecoder attempts to decode a stream of JSON or YAML documents.
// While JSON is YAML, the way Go's JSON decode defines a multi-document stream
// is a series of JSON objects (e.g. {}{}), but YAML defines a multi-document
// stream as a series of documents separated by "---".
//
// This decoder will attempt to decode the stream as JSON first, and if that
// fails, it will switch to YAML. Once it determines the stream is JSON (by
// finding a non-YAML-delimited series of objects), it will not switch to YAML.
// Once it switches to YAML it will not switch back to JSON.
type YAMLOrJSONDecoder struct {
	json         *json.Decoder
	jsonConsumed int64 // of the stream total, how much was JSON?
	yaml         *YAMLToJSONDecoder
	yamlConsumed int64 // of the stream total, how much was YAML?
	stream       *StreamReader
	count        int // how many objects have been decoded
}

type JSONSyntaxError struct {
	Offset int64
	Err    error
}

func (e JSONSyntaxError) Error() string {
	return fmt.Sprintf("json: offset %d: %s", e.Offset, e.Err.Error())
}

type YAMLSyntaxError struct {
	err error
}

func (e YAMLSyntaxError) Error() string {
	return e.err.Error()
}

// NewYAMLOrJSONDecoder returns a decoder that will process YAML documents
// or JSON documents from the given reader as a stream. bufferSize determines
// how far into the stream the decoder will look to figure out whether this
// is a JSON stream (has whitespace followed by an open brace).
func NewYAMLOrJSONDecoder(r io.Reader, bufferSize int) *YAMLOrJSONDecoder {
	d := &YAMLOrJSONDecoder{}

	reader, _, mightBeJSON := GuessJSONStream(r, bufferSize)
	d.stream = reader
	if mightBeJSON {
		d.json = json.NewDecoder(reader)
	} else {
		d.yaml = NewYAMLToJSONDecoder(reader)
	}
	return d
}

// Decode unmarshals the next object from the underlying stream into the
// provide object, or returns an error.
func (d *YAMLOrJSONDecoder) Decode(into interface{}) error {
	// Because we don't know if this is a JSON or YAML stream, a failure from
	// both decoders is ambiguous. When in doubt, it will return the error from
	// the JSON decoder.  Unfortunately, this means that if the first document
	// is invalid YAML, the error won't be awesome.
	// TODO: the errors from YAML are not great, we could improve them a lot.
	var firstErr error
	if d.json != nil {
		err := d.json.Decode(into)
		if err == nil {
			d.count++
			consumed := d.json.InputOffset() - d.jsonConsumed
			d.stream.Consume(int(consumed))
			d.jsonConsumed += consumed
			return nil
		}
		if err == io.EOF { //nolint:errorlint
			return err
		}
		var syntax *json.SyntaxError
		if ok := errors.As(err, &syntax); ok {
			firstErr = JSONSyntaxError{
				Offset: syntax.Offset,
				Err:    syntax,
			}
		} else {
			firstErr = err
		}
		if d.count > 1 {
			// If we found 0 or 1 JSON object(s), this stream is still
			// ambiguous.  But if we found more than 1 JSON object, then this
			// is an unambiguous JSON stream, and we should not switch to YAML.
			return err
		}
		// If JSON decoding hits the end of one object and then fails on the
		// next, it leaves any leading whitespace in the buffer, which can
		// confuse the YAML decoder. We just eat any whitespace we find, up to
		// and including the first newline.
		d.stream.Rewind()
		if err := d.consumeWhitespace(); err == nil {
			d.yaml = NewYAMLToJSONDecoder(d.stream)
		}
		d.json = nil
	}
	if d.yaml != nil {
		err := d.yaml.Decode(into)
		if err == nil {
			consumed := int64(d.yaml.InputOffset()) - d.yamlConsumed
			d.stream.Consume(int(consumed))
			d.yamlConsumed += consumed
			d.count++
			return nil
		}
		if err == io.EOF { //nolint:errorlint
			return err
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return firstErr
	}
	return fmt.Errorf("decoding failed as both JSON and YAML")
}

func (d *YAMLOrJSONDecoder) consumeWhitespace() error {
	consumed := 0
	for {
		buf, err := d.stream.ReadN(4)
		if err != nil && err == io.EOF { //nolint:errorlint
			return err
		}
		r, sz := utf8.DecodeRune(buf)
		if r == utf8.RuneError || sz == 0 {
			return fmt.Errorf("invalid utf8 rune")
		}
		d.stream.RewindN(len(buf) - sz)
		if !unicode.IsSpace(r) {
			d.stream.RewindN(sz)
			d.stream.Consume(consumed)
			return nil
		}
		if r == '\n' {
			d.stream.Consume(consumed)
			return nil
		}
		if err == io.EOF { //nolint:errorlint
			break
		}
		consumed += sz
	}
	return io.EOF
}

type Reader interface {
	Read() ([]byte, error)
}

type YAMLReader struct {
	reader Reader
}

func NewYAMLReader(r *bufio.Reader) *YAMLReader {
	return &YAMLReader{
		reader: &LineReader{reader: r},
	}
}

// Read returns a full YAML document.
func (r *YAMLReader) Read() ([]byte, error) {
	var buffer bytes.Buffer
	for {
		line, err := r.reader.Read()
		if err != nil && err != io.EOF { //nolint:errorlint
			return nil, err
		}

		sep := len([]byte(separator))
		if i := bytes.Index(line, []byte(separator)); i == 0 {
			// We have a potential document terminator
			i += sep
			trimmed := strings.TrimSpace(string(line[i:]))
			// We only allow comments and spaces following the yaml doc separator, otherwise we'll return an error
			if len(trimmed) > 0 && string(trimmed[0]) != "#" {
				return nil, YAMLSyntaxError{
					err: fmt.Errorf("invalid Yaml document separator: %s", trimmed),
				}
			}
			if buffer.Len() != 0 {
				return buffer.Bytes(), nil
			}
			if err == io.EOF { //nolint:errorlint
				return nil, err
			}
		}
		if err == io.EOF { //nolint:errorlint
			if buffer.Len() != 0 {
				// If we're at EOF, we have a final, non-terminated line. Return it.
				return buffer.Bytes(), nil
			}
			return nil, err
		}
		buffer.Write(line)
	}
}

type LineReader struct {
	reader *bufio.Reader
}

// Read returns a single line (with '\n' ended) from the underlying reader.
// An error is returned iff there is an error with the underlying reader.
func (r *LineReader) Read() ([]byte, error) {
	var (
		isPrefix bool  = true
		err      error = nil
		line     []byte
		buffer   bytes.Buffer
	)

	for isPrefix && err == nil {
		line, isPrefix, err = r.reader.ReadLine()
		buffer.Write(line)
	}
	buffer.WriteByte('\n')
	return buffer.Bytes(), err
}

// GuessJSONStream scans the provided reader up to size, looking
// for an open brace indicating this is JSON. It will return the
// bufio.Reader it creates for the consumer.
func GuessJSONStream(r io.Reader, size int) (*StreamReader, []byte, bool) {
	buffer := NewStreamReader(r, size)
	b, _ := buffer.Peek(size)
	return buffer, b, IsJSONBuffer(b)
}

// IsJSONBuffer scans the provided buffer, looking
// for an open brace indicating this is JSON.
func IsJSONBuffer(buf []byte) bool {
	return hasPrefix(buf, jsonPrefix)
}

var jsonPrefix = []byte("{")

// Return true if the first non-whitespace bytes in buf is
// prefix.
func hasPrefix(buf []byte, prefix []byte) bool {
	trim := bytes.TrimLeftFunc(buf, unicode.IsSpace)
	return bytes.HasPrefix(trim, prefix)
}
