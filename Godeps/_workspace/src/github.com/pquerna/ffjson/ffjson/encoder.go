package ffjson

/**
 *  Copyright 2015 Paul Querna, Klaus Post
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

import (
	"encoding/json"
	"errors"
	fflib "github.com/pquerna/ffjson/fflib/v1"
	"io"
	"reflect"
)

// This is a reusable encoder.
// It allows to encode many objects to a single writer.
// This should not be used by more than one goroutine at the time.
type Encoder struct {
	buf fflib.Buffer
	w   io.Writer
	enc *json.Encoder
}

// NewEncoder returns a reusable Encoder.
// Output will be written to the supplied writer.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w, enc: json.NewEncoder(w)}
}

// Encode the data in the supplied value to the stream
// given on creation.
// When the function returns the output has been
// written to the stream.
func (e *Encoder) Encode(v interface{}) error {
	f, ok := v.(marshalerFaster)
	if ok {
		e.buf.Reset()
		err := f.MarshalJSONBuf(&e.buf)
		if err != nil {
			return err
		}

		_, err = io.Copy(e.w, &e.buf)
		return err
	}

	return e.enc.Encode(v)
}

// EncodeFast will unmarshal the data if fast marshall is available.
// This function can be used if you want to be sure the fast
// marshal is used or in testing.
// If you would like to have fallback to encoding/json you can use the
// regular Encode() method.
func (e *Encoder) EncodeFast(v interface{}) error {
	_, ok := v.(marshalerFaster)
	if !ok {
		return errors.New("ffjson marshal not available for type " + reflect.TypeOf(v).String())
	}
	return e.Encode(v)
}
