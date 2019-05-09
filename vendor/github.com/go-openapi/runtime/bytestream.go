// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runtime

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"reflect"

	"github.com/go-openapi/swag"
)

func defaultCloser() error { return nil }

type byteStreamOpt func(opts *byteStreamOpts)

// ClosesStream when the bytestream consumer or producer is finished
func ClosesStream(opts *byteStreamOpts) {
	opts.Close = true
}

type byteStreamOpts struct {
	Close bool
}

// ByteStreamConsumer creates a consmer for byte streams,
// takes a Writer/BinaryUnmarshaler interface or binary slice by reference,
// and reads from the provided reader
func ByteStreamConsumer(opts ...byteStreamOpt) Consumer {
	var vals byteStreamOpts
	for _, opt := range opts {
		opt(&vals)
	}

	return ConsumerFunc(func(reader io.Reader, data interface{}) error {
		if reader == nil {
			return errors.New("ByteStreamConsumer requires a reader") // early exit
		}

		close := defaultCloser
		if vals.Close {
			if cl, ok := reader.(io.Closer); ok {
				close = cl.Close
			}
		}
		defer close()

		if wrtr, ok := data.(io.Writer); ok {
			_, err := io.Copy(wrtr, reader)
			return err
		}

		buf := new(bytes.Buffer)
		_, err := buf.ReadFrom(reader)
		if err != nil {
			return err
		}
		b := buf.Bytes()

		if bu, ok := data.(encoding.BinaryUnmarshaler); ok {
			return bu.UnmarshalBinary(b)
		}

		if t := reflect.TypeOf(data); data != nil && t.Kind() == reflect.Ptr {
			v := reflect.Indirect(reflect.ValueOf(data))
			if t = v.Type(); t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
				v.SetBytes(b)
				return nil
			}
		}

		return fmt.Errorf("%v (%T) is not supported by the ByteStreamConsumer, %s",
			data, data, "can be resolved by supporting Writer/BinaryUnmarshaler interface")
	})
}

// ByteStreamProducer creates a producer for byte streams,
// takes a Reader/BinaryMarshaler interface or binary slice,
// and writes to a writer (essentially a pipe)
func ByteStreamProducer(opts ...byteStreamOpt) Producer {
	var vals byteStreamOpts
	for _, opt := range opts {
		opt(&vals)
	}
	return ProducerFunc(func(writer io.Writer, data interface{}) error {
		if writer == nil {
			return errors.New("ByteStreamProducer requires a writer") // early exit
		}
		close := defaultCloser
		if vals.Close {
			if cl, ok := writer.(io.Closer); ok {
				close = cl.Close
			}
		}
		defer close()

		if rdr, ok := data.(io.Reader); ok {
			_, err := io.Copy(writer, rdr)
			return err
		}

		if bm, ok := data.(encoding.BinaryMarshaler); ok {
			bytes, err := bm.MarshalBinary()
			if err != nil {
				return err
			}

			_, err = writer.Write(bytes)
			return err
		}

		if data != nil {
			if e, ok := data.(error); ok {
				_, err := writer.Write([]byte(e.Error()))
				return err
			}

			v := reflect.Indirect(reflect.ValueOf(data))
			if t := v.Type(); t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
				_, err := writer.Write(v.Bytes())
				return err
			}
			if t := v.Type(); t.Kind() == reflect.Struct || t.Kind() == reflect.Slice {
				b, err := swag.WriteJSON(data)
				if err != nil {
					return err
				}
				_, err = writer.Write(b)
				return err
			}
		}

		return fmt.Errorf("%v (%T) is not supported by the ByteStreamProducer, %s",
			data, data, "can be resolved by supporting Reader/BinaryMarshaler interface")
	})
}
