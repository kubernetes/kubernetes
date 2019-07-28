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

// TextConsumer creates a new text consumer
func TextConsumer() Consumer {
	return ConsumerFunc(func(reader io.Reader, data interface{}) error {
		if reader == nil {
			return errors.New("TextConsumer requires a reader") // early exit
		}

		buf := new(bytes.Buffer)
		_, err := buf.ReadFrom(reader)
		if err != nil {
			return err
		}
		b := buf.Bytes()

		if tu, ok := data.(encoding.TextUnmarshaler); ok {
			err := tu.UnmarshalText(b)
			if err != nil {
				return fmt.Errorf("text consumer: %v", err)
			}

			return nil
		}

		t := reflect.TypeOf(data)
		if data != nil && t.Kind() == reflect.Ptr {
			v := reflect.Indirect(reflect.ValueOf(data))
			if t.Elem().Kind() == reflect.String {
				v.SetString(string(b))
				return nil
			}
		}

		return fmt.Errorf("%v (%T) is not supported by the TextConsumer, %s",
			data, data, "can be resolved by supporting TextUnmarshaler interface")
	})
}

// TextProducer creates a new text producer
func TextProducer() Producer {
	return ProducerFunc(func(writer io.Writer, data interface{}) error {
		if writer == nil {
			return errors.New("TextProducer requires a writer") // early exit
		}

		if data == nil {
			return errors.New("no data given to produce text from")
		}

		if tm, ok := data.(encoding.TextMarshaler); ok {
			txt, err := tm.MarshalText()
			if err != nil {
				return fmt.Errorf("text producer: %v", err)
			}
			_, err = writer.Write(txt)
			return err
		}

		if str, ok := data.(error); ok {
			_, err := writer.Write([]byte(str.Error()))
			return err
		}

		if str, ok := data.(fmt.Stringer); ok {
			_, err := writer.Write([]byte(str.String()))
			return err
		}

		v := reflect.Indirect(reflect.ValueOf(data))
		if t := v.Type(); t.Kind() == reflect.Struct || t.Kind() == reflect.Slice {
			b, err := swag.WriteJSON(data)
			if err != nil {
				return err
			}
			_, err = writer.Write(b)
			return err
		}
		if v.Kind() != reflect.String {
			return fmt.Errorf("%T is not a supported type by the TextProducer", data)
		}

		_, err := writer.Write([]byte(v.String()))
		return err
	})
}
