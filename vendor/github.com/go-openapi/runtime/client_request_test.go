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
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
)

type trw struct {
	Headers http.Header
	Body    interface{}
}

func (t *trw) SetHeaderParam(name string, values ...string) error {
	if t.Headers == nil {
		t.Headers = make(http.Header)
	}
	t.Headers.Set(name, values[0])
	return nil
}

func (t *trw) SetQueryParam(_ string, _ ...string) error { return nil }

func (t *trw) SetFormParam(_ string, _ ...string) error { return nil }

func (t *trw) SetPathParam(_ string, _ string) error { return nil }

func (t *trw) SetFileParam(_ string, _ *os.File) error { return nil }

func (t *trw) SetBodyParam(body interface{}) error {
	t.Body = body
	return nil
}

func (t *trw) SetTimeout(timeout time.Duration) error {
	return nil
}

func TestRequestWriterFunc(t *testing.T) {

	hand := ClientRequestWriterFunc(func(r ClientRequest, reg strfmt.Registry) error {
		r.SetHeaderParam("blah", "blah blah")
		r.SetBodyParam(struct{ Name string }{"Adriana"})
		return nil
	})

	tr := new(trw)
	hand.WriteToRequest(tr, nil)
	assert.Equal(t, "blah blah", tr.Headers.Get("blah"))
	assert.Equal(t, "Adriana", tr.Body.(struct{ Name string }).Name)
}
