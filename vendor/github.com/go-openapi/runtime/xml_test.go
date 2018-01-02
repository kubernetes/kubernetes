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
	"encoding/xml"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

var consProdXML = `<person><name>Somebody</name><id>1</id></person>`

func TestXMLConsumer(t *testing.T) {
	cons := XMLConsumer()
	var data struct {
		XMLName xml.Name `xml:"person"`
		Name    string   `xml:"name"`
		ID      int      `xml:"id"`
	}
	err := cons.Consume(bytes.NewBuffer([]byte(consProdXML)), &data)
	assert.NoError(t, err)
	assert.Equal(t, "Somebody", data.Name)
	assert.Equal(t, 1, data.ID)
}

func TestXMLProducer(t *testing.T) {
	prod := XMLProducer()
	data := struct {
		XMLName xml.Name `xml:"person"`
		Name    string   `xml:"name"`
		ID      int      `xml:"id"`
	}{Name: "Somebody", ID: 1}

	rw := httptest.NewRecorder()
	err := prod.Produce(rw, data)
	assert.NoError(t, err)
	assert.Equal(t, consProdXML, rw.Body.String())
}
