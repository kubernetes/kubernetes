/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"unicode"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

// ToJSON converts a single YAML document into a JSON document
// or returns an error. If the document appears to be JSON the
// YAML decoding path is not used (so that error messages are
// JSON specific).
func ToJSON(data []byte) ([]byte, error) {
	if hasJSONPrefix(data) {
		return data, nil
	}
	return yaml.YAMLToJSON(data)
}

// YAMLToJSONDecoder decodes YAML documents from an io.Reader by
// separating individual documents. It first converts the YAML
// body to JSON, then unmarshals the JSON.
type YAMLToJSONDecoder struct {
	scanner *bufio.Scanner
}

// NewYAMLToJSONDecoder decodes YAML documents from the provided
// stream in chunks by converting each document (as defined by
// the YAML spec) into its own chunk, converting it to JSON via
// yaml.YAMLToJSON, and then passing it to json.Decoder.
func NewYAMLToJSONDecoder(r io.Reader) *YAMLToJSONDecoder {
	scanner := bufio.NewScanner(r)
	scanner.Split(splitYAMLDocument)
	return &YAMLToJSONDecoder{
		scanner: scanner,
	}
}

// Decode reads a YAML document as JSON from the stream or returns
// an error. The decoding rules match json.Unmarshal, not
// yaml.Unmarshal.
func (d *YAMLToJSONDecoder) Decode(into interface{}) error {
	if d.scanner.Scan() {
		data, err := yaml.YAMLToJSON(d.scanner.Bytes())
		if err != nil {
			return err
		}
		return json.Unmarshal(data, into)
	}
	err := d.scanner.Err()
	if err == nil {
		err = io.EOF
	}
	return err
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
		return len(d.remaining), nil
	}

	// caller will need to reread
	copy(data, d.remaining[:left])
	d.remaining = d.remaining[left:]
	return len(data), io.ErrShortBuffer
}

func (d *YAMLDecoder) Close() error {
	return d.r.Close()
}

const yamlSeparator = "\n---"

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

// decoder is a convenience interface for Decode.
type decoder interface {
	Decode(into interface{}) error
}

// YAMLOrJSONDecoder attempts to decode a stream of JSON documents or
// YAML documents by sniffing for a leading { character.
type YAMLOrJSONDecoder struct {
	r          io.Reader
	bufferSize int

	decoder decoder
}

// NewYAMLOrJSONDecoder returns a decoder that will process YAML documents
// or JSON documents from the given reader as a stream. bufferSize determines
// how far into the stream the decoder will look to figure out whether this
// is a JSON stream (has whitespace followed by an open brace).
func NewYAMLOrJSONDecoder(r io.Reader, bufferSize int) *YAMLOrJSONDecoder {
	return &YAMLOrJSONDecoder{
		r:          r,
		bufferSize: bufferSize,
	}
}

// Decode unmarshals the next object from the underlying stream into the
// provide object, or returns an error.
func (d *YAMLOrJSONDecoder) Decode(into interface{}) error {
	if d.decoder == nil {
		buffer, isJSON := GuessJSONStream(d.r, d.bufferSize)
		if isJSON {
			glog.V(4).Infof("decoding stream as JSON")
			d.decoder = json.NewDecoder(buffer)
		} else {
			glog.V(4).Infof("decoding stream as YAML")
			d.decoder = NewYAMLToJSONDecoder(buffer)
		}
	}
	err := d.decoder.Decode(into)
	if jsonDecoder, ok := d.decoder.(*json.Decoder); ok {
		if syntax, ok := err.(*json.SyntaxError); ok {
			data, readErr := ioutil.ReadAll(jsonDecoder.Buffered())
			if readErr != nil {
				glog.V(4).Infof("reading stream failed: %v", readErr)
			}
			js := string(data)
			start := strings.LastIndex(js[:syntax.Offset], "\n") + 1
			line := strings.Count(js[:start], "\n")
			return fmt.Errorf("json: line %d: %s", line, syntax.Error())
		}
	}
	return err
}

// GuessJSONStream scans the provided reader up to size, looking
// for an open brace indicating this is JSON. It will return the
// bufio.Reader it creates for the consumer.
func GuessJSONStream(r io.Reader, size int) (io.Reader, bool) {
	buffer := bufio.NewReaderSize(r, size)
	b, _ := buffer.Peek(size)
	return buffer, hasJSONPrefix(b)
}

var jsonPrefix = []byte("{")

// hasJSONPrefix returns true if the provided buffer appears to start with
// a JSON open brace.
func hasJSONPrefix(buf []byte) bool {
	return hasPrefix(buf, jsonPrefix)
}

// Return true if the first non-whitespace bytes in buf is
// prefix.
func hasPrefix(buf []byte, prefix []byte) bool {
	trim := bytes.TrimLeftFunc(buf, unicode.IsSpace)
	return bytes.HasPrefix(trim, prefix)
}
