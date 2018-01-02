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
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

var consProdText = `The quick brown fox jumped over the lazy dog.`

func TestTextConsumer(t *testing.T) {
	cons := TextConsumer()
	var data string
	err := cons.Consume(bytes.NewBuffer([]byte(consProdText)), &data)
	assert.NoError(t, err)
	assert.Equal(t, consProdText, data)
}

func TestTextProducer(t *testing.T) {
	prod := TextProducer()
	rw := httptest.NewRecorder()
	err := prod.Produce(rw, consProdText)
	assert.NoError(t, err)
	assert.Equal(t, consProdText, rw.Body.String())
	rw2 := httptest.NewRecorder()
	err2 := prod.Produce(rw2, &consProdText)
	assert.NoError(t, err2)
	assert.Equal(t, consProdText, rw2.Body.String())
}
