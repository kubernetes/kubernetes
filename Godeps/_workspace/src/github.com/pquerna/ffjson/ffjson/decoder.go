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
	"io/ioutil"
	"reflect"
)

// This is a reusable decoder.
// This should not be used by more than one goroutine at the time.
type Decoder struct {
	fs *fflib.FFLexer
}

// NewDecoder returns a reusable Decoder.
func NewDecoder() *Decoder {
	return &Decoder{}
}

// Decode the data in the supplied data slice.
func (d *Decoder) Decode(data []byte, v interface{}) error {
	f, ok := v.(unmarshalFaster)
	if ok {
		if d.fs == nil {
			d.fs = fflib.NewFFLexer(data)
		} else {
			d.fs.Reset(data)
		}
		return f.UnmarshalJSONFFLexer(d.fs, fflib.FFParse_map_start)
	}

	um, ok := v.(json.Unmarshaler)
	if ok {
		return um.UnmarshalJSON(data)
	}
	return json.Unmarshal(data, v)
}

// Decode the data from the supplied reader.
// You should expect that data is read into memory before it is decoded.
func (d *Decoder) DecodeReader(r io.Reader, v interface{}) error {
	_, ok := v.(unmarshalFaster)
	_, ok2 := v.(json.Unmarshaler)
	if ok || ok2 {
		data, err := ioutil.ReadAll(r)
		if err != nil {
			return err
		}
		defer fflib.Pool(data)
		return d.Decode(data, v)
	}
	dec := json.NewDecoder(r)
	return dec.Decode(v)
}

// DecodeFast will unmarshal the data if fast unmarshal is available.
// This function can be used if you want to be sure the fast
// unmarshal is used or in testing.
// If you would like to have fallback to encoding/json you can use the
// regular Decode() method.
func (d *Decoder) DecodeFast(data []byte, v interface{}) error {
	f, ok := v.(unmarshalFaster)
	if !ok {
		return errors.New("ffjson unmarshal not available for type " + reflect.TypeOf(v).String())
	}
	if d.fs == nil {
		d.fs = fflib.NewFFLexer(data)
	} else {
		d.fs.Reset(data)
	}
	return f.UnmarshalJSONFFLexer(d.fs, fflib.FFParse_map_start)
}
