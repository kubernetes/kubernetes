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

package swag

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

type testNameStruct struct {
	Name       string `json:"name"`
	NotTheSame int64  `json:"plain"`
	Ignored    string `json:"-"`
}

func TestNameProvider(t *testing.T) {

	provider := NewNameProvider()

	var obj = testNameStruct{}

	nm, ok := provider.GetGoName(obj, "name")
	assert.True(t, ok)
	assert.Equal(t, "Name", nm)

	nm, ok = provider.GetGoName(obj, "plain")
	assert.True(t, ok)
	assert.Equal(t, "NotTheSame", nm)

	nm, ok = provider.GetGoName(obj, "doesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetGoName(obj, "ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	tpe := reflect.TypeOf(obj)
	nm, ok = provider.GetGoNameForType(tpe, "name")
	assert.True(t, ok)
	assert.Equal(t, "Name", nm)

	nm, ok = provider.GetGoNameForType(tpe, "plain")
	assert.True(t, ok)
	assert.Equal(t, "NotTheSame", nm)

	nm, ok = provider.GetGoNameForType(tpe, "doesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetGoNameForType(tpe, "ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	ptr := &obj
	nm, ok = provider.GetGoName(ptr, "name")
	assert.True(t, ok)
	assert.Equal(t, "Name", nm)

	nm, ok = provider.GetGoName(ptr, "plain")
	assert.True(t, ok)
	assert.Equal(t, "NotTheSame", nm)

	nm, ok = provider.GetGoName(ptr, "doesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetGoName(ptr, "ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONName(obj, "Name")
	assert.True(t, ok)
	assert.Equal(t, "name", nm)

	nm, ok = provider.GetJSONName(obj, "NotTheSame")
	assert.True(t, ok)
	assert.Equal(t, "plain", nm)

	nm, ok = provider.GetJSONName(obj, "DoesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONName(obj, "Ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONNameForType(tpe, "Name")
	assert.True(t, ok)
	assert.Equal(t, "name", nm)

	nm, ok = provider.GetJSONNameForType(tpe, "NotTheSame")
	assert.True(t, ok)
	assert.Equal(t, "plain", nm)

	nm, ok = provider.GetJSONNameForType(tpe, "doesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONNameForType(tpe, "Ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONName(ptr, "Name")
	assert.True(t, ok)
	assert.Equal(t, "name", nm)

	nm, ok = provider.GetJSONName(ptr, "NotTheSame")
	assert.True(t, ok)
	assert.Equal(t, "plain", nm)

	nm, ok = provider.GetJSONName(ptr, "doesNotExist")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nm, ok = provider.GetJSONName(ptr, "Ignored")
	assert.False(t, ok)
	assert.Empty(t, nm)

	nms := provider.GetJSONNames(ptr)
	assert.Len(t, nms, 2)

	assert.Len(t, provider.index, 1)

}

func TestJSONConcatenation(t *testing.T) {
	assert.Nil(t, ConcatJSON())
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`)), []byte(`{"id":1}`))
	assert.Equal(t, ConcatJSON([]byte(`{}`), []byte(`{}`)), []byte(`{}`))
	assert.Equal(t, ConcatJSON([]byte(`[]`), []byte(`[]`)), []byte(`[]`))
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`), []byte(`{"name":"Rachel"}`)), []byte(`{"id":1,"name":"Rachel"}`))
	assert.Equal(t, ConcatJSON([]byte(`[{"id":1}]`), []byte(`[{"name":"Rachel"}]`)), []byte(`[{"id":1},{"name":"Rachel"}]`))
	assert.Equal(t, ConcatJSON([]byte(`{}`), []byte(`{"name":"Rachel"}`)), []byte(`{"name":"Rachel"}`))
	assert.Equal(t, ConcatJSON([]byte(`[]`), []byte(`[{"name":"Rachel"}]`)), []byte(`[{"name":"Rachel"}]`))
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`), []byte(`{}`)), []byte(`{"id":1}`))
	assert.Equal(t, ConcatJSON([]byte(`[{"id":1}]`), []byte(`[]`)), []byte(`[{"id":1}]`))
	assert.Equal(t, ConcatJSON([]byte(`{}`), []byte(`{}`), []byte(`{}`)), []byte(`{}`))
	assert.Equal(t, ConcatJSON([]byte(`[]`), []byte(`[]`), []byte(`[]`)), []byte(`[]`))
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`), []byte(`{"name":"Rachel"}`), []byte(`{"age":32}`)), []byte(`{"id":1,"name":"Rachel","age":32}`))
	assert.Equal(t, ConcatJSON([]byte(`[{"id":1}]`), []byte(`[{"name":"Rachel"}]`), []byte(`[{"age":32}]`)), []byte(`[{"id":1},{"name":"Rachel"},{"age":32}]`))
	assert.Equal(t, ConcatJSON([]byte(`{}`), []byte(`{"name":"Rachel"}`), []byte(`{"age":32}`)), []byte(`{"name":"Rachel","age":32}`))
	assert.Equal(t, ConcatJSON([]byte(`[]`), []byte(`[{"name":"Rachel"}]`), []byte(`[{"age":32}]`)), []byte(`[{"name":"Rachel"},{"age":32}]`))
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`), []byte(`{}`), []byte(`{"age":32}`)), []byte(`{"id":1,"age":32}`))
	assert.Equal(t, ConcatJSON([]byte(`[{"id":1}]`), []byte(`[]`), []byte(`[{"age":32}]`)), []byte(`[{"id":1},{"age":32}]`))
	assert.Equal(t, ConcatJSON([]byte(`{"id":1}`), []byte(`{"name":"Rachel"}`), []byte(`{}`)), []byte(`{"id":1,"name":"Rachel"}`))
	assert.Equal(t, ConcatJSON([]byte(`[{"id":1}]`), []byte(`[{"name":"Rachel"}]`), []byte(`[]`)), []byte(`[{"id":1},{"name":"Rachel"}]`))

}
