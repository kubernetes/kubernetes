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
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

type response struct {
}

func (r response) Code() int {
	return 490
}
func (r response) Message() string {
	return "the message"
}
func (r response) GetHeader(_ string) string {
	return "the header"
}
func (r response) Body() io.ReadCloser {
	return ioutil.NopCloser(bytes.NewBufferString("the content"))
}

func TestResponseReaderFunc(t *testing.T) {
	var actual struct {
		Header, Message, Body string
		Code                  int
	}
	reader := ClientResponseReaderFunc(func(r ClientResponse, _ Consumer) (interface{}, error) {
		b, _ := ioutil.ReadAll(r.Body())
		actual.Body = string(b)
		actual.Code = r.Code()
		actual.Message = r.Message()
		actual.Header = r.GetHeader("blah")
		return actual, nil
	})
	reader.ReadResponse(response{}, nil)
	assert.Equal(t, "the content", actual.Body)
	assert.Equal(t, "the message", actual.Message)
	assert.Equal(t, "the header", actual.Header)
	assert.Equal(t, 490, actual.Code)
}
