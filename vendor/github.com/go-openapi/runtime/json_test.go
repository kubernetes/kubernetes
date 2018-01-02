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
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

var consProdJSON = `{"name":"Somebody","id":1}`

type eofRdr struct {
}

func (r *eofRdr) Read(d []byte) (int, error) {
	return 0, io.EOF
}

func TestJSONConsumer(t *testing.T) {
	cons := JSONConsumer()
	var data struct {
		Name string
		ID   int
	}
	err := cons.Consume(bytes.NewBuffer([]byte(consProdJSON)), &data)
	if assert.NoError(t, err) {
		assert.Equal(t, "Somebody", data.Name)
		assert.Equal(t, 1, data.ID)

		err = cons.Consume(new(eofRdr), &data)
		assert.Error(t, err)
	}
}

func TestJSONProducer(t *testing.T) {
	prod := JSONProducer()
	data := struct {
		Name string `json:"name"`
		ID   int    `json:"id"`
	}{Name: "Somebody", ID: 1}

	rw := httptest.NewRecorder()
	err := prod.Produce(rw, data)
	assert.NoError(t, err)
	assert.Equal(t, consProdJSON+"\n", rw.Body.String())
}
