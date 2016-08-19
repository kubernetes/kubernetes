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
	"errors"
	"io"
)

// ByteStreamConsumer creates a consmer for byte streams, takes a writer and reads from the provided reader
func ByteStreamConsumer() Consumer {
	return ConsumerFunc(func(r io.Reader, v interface{}) error {
		wrtr, ok := v.(io.Writer)
		if !ok {
			return errors.New("ByteStreamConsumer can only deal with io.Writer")
		}

		_, err := io.Copy(wrtr, r)
		return err
	})
}

// ByteStreamProducer creates a producer for byte streams, takes a reader, writes to a writer (essentially a pipe)
func ByteStreamProducer() Producer {
	return ProducerFunc(func(w io.Writer, v interface{}) error {
		rdr, ok := v.(io.Reader)
		if !ok {
			return errors.New("ByteStreamProducer can only deal with io.Reader")
		}
		_, err := io.Copy(w, rdr)
		return err
	})
}
