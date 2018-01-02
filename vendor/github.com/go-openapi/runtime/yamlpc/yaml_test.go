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

package yamlpc

import (
	"bytes"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

var consProdYAML = "name: Somebody\nid: 1\n"

func TestYAMLConsumer(t *testing.T) {
	cons := YAMLConsumer()
	var data struct {
		Name string
		ID   int
	}
	err := cons.Consume(bytes.NewBuffer([]byte(consProdYAML)), &data)
	assert.NoError(t, err)
	assert.Equal(t, "Somebody", data.Name)
	assert.Equal(t, 1, data.ID)
}

func TestYAMLProducer(t *testing.T) {
	prod := YAMLProducer()
	data := struct {
		Name string `yaml:"name"`
		ID   int    `yaml:"id"`
	}{Name: "Somebody", ID: 1}

	rw := httptest.NewRecorder()
	err := prod.Produce(rw, data)
	assert.NoError(t, err)
	assert.Equal(t, consProdYAML, rw.Body.String())
}

type failReader struct {
}

func (f *failReader) Read(p []byte) (n int, err error) {
	return 0, errors.New("expected")
}
func TestFailYAMLReader(t *testing.T) {
	cons := YAMLConsumer()
	assert.Error(t, cons.Consume(&failReader{}, nil))
}
