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
	"io"
	"unsafe"

	"github.com/go-openapi/swag"
)

// TextConsumer creates a new text consumer
func TextConsumer() Consumer {
	return ConsumerFunc(func(reader io.Reader, data interface{}) error {
		buf := new(bytes.Buffer)
		_, err := buf.ReadFrom(reader)
		if err != nil {
			return err
		}
		b := buf.Bytes()
		*(data.(*string)) = *(*string)(unsafe.Pointer(&b))
		return nil
	})
}

// TextProducer creates a new text producer
func TextProducer() Producer {
	return ProducerFunc(func(writer io.Writer, data interface{}) error {
		var buf *bytes.Buffer
		switch tped := data.(type) {
		case *string:
			buf = bytes.NewBufferString(swag.StringValue(tped))
		case string:
			buf = bytes.NewBufferString(tped)
		}
		_, err := buf.WriteTo(writer)
		return err
	})
}
