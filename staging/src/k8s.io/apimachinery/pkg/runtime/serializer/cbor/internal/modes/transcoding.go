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
	"encoding/json"
	"errors"
	"io"

	kjson "sigs.k8s.io/json"
)

type TranscodeFunc func(dst io.Writer, src io.Reader) error

func (f TranscodeFunc) Transcode(dst io.Writer, src io.Reader) error {
	return f(dst, src)
}

func TranscodeFromJSON(dst io.Writer, src io.Reader) error {
	var tmp any
	dec := kjson.NewDecoderCaseSensitivePreserveInts(src)
	if err := dec.Decode(&tmp); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("extraneous data")
	}

	return encode.MarshalTo(tmp, dst)
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
