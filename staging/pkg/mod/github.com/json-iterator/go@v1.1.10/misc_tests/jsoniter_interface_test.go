package misc_tests

import (
	"encoding/json"
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func Test_nil_non_empty_interface(t *testing.T) {
	type TestObject struct {
		Field []io.Closer
	}
	should := require.New(t)
	obj := TestObject{}
	b := []byte(`{"Field":["AAA"]}`)
	should.NotNil(json.Unmarshal(b, &obj))
	should.NotNil(jsoniter.Unmarshal(b, &obj))
}

func Test_nil_out_null_interface(t *testing.T) {
	type TestData struct {
		Field interface{} `json:"field"`
	}
	should := require.New(t)

	var boolVar bool
	obj := TestData{
		Field: &boolVar,
	}

	data1 := []byte(`{"field": true}`)

	err := jsoniter.Unmarshal(data1, &obj)
	should.NoError(err)
	should.Equal(true, *(obj.Field.(*bool)))

	data2 := []byte(`{"field": null}`)

	err = jsoniter.Unmarshal(data2, &obj)
	should.NoError(err)
	should.Nil(obj.Field)

	// Checking stdlib behavior matches.
	obj2 := TestData{
		Field: &boolVar,
	}

	err = json.Unmarshal(data1, &obj2)
	should.NoError(err)
	should.Equal(true, *(obj2.Field.(*bool)))

	err = json.Unmarshal(data2, &obj2)
	should.NoError(err)
	should.Equal(nil, obj2.Field)
}

func Test_overwrite_interface_ptr_value_with_nil(t *testing.T) {
	type Wrapper struct {
		Payload interface{} `json:"payload,omitempty"`
	}
	type Payload struct {
		Value int `json:"val,omitempty"`
	}

	should := require.New(t)

	payload := &Payload{}
	wrapper := &Wrapper{
		Payload: &payload,
	}

	err := json.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.NoError(err)
	should.Equal(&payload, wrapper.Payload)
	should.Equal(42, (*(wrapper.Payload.(**Payload))).Value)

	err = json.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.NoError(err)
	should.Equal(&payload, wrapper.Payload)
	should.Equal((*Payload)(nil), payload)

	payload = &Payload{}
	wrapper = &Wrapper{
		Payload: &payload,
	}

	err = jsoniter.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.Equal(nil, err)
	should.Equal(&payload, wrapper.Payload)
	should.Equal(42, (*(wrapper.Payload.(**Payload))).Value)

	err = jsoniter.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.NoError(err)
	should.Equal(&payload, wrapper.Payload)
	should.Equal((*Payload)(nil), payload)
}

func Test_overwrite_interface_value_with_nil(t *testing.T) {
	type Wrapper struct {
		Payload interface{} `json:"payload,omitempty"`
	}
	type Payload struct {
		Value int `json:"val,omitempty"`
	}

	should := require.New(t)

	payload := &Payload{}
	wrapper := &Wrapper{
		Payload: payload,
	}

	err := json.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.NoError(err)
	should.Equal(42, wrapper.Payload.(*Payload).Value)

	err = json.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.NoError(err)
	should.Equal(nil, wrapper.Payload)
	should.Equal(42, payload.Value)

	payload = &Payload{}
	wrapper = &Wrapper{
		Payload: payload,
	}

	err = jsoniter.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.Equal(nil, err)
	should.Equal(42, wrapper.Payload.(*Payload).Value)

	err = jsoniter.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.Equal(nil, err)
	should.Equal(nil, wrapper.Payload)
	should.Equal(42, payload.Value)
}

func Test_unmarshal_into_nil(t *testing.T) {
	type Payload struct {
		Value int `json:"val,omitempty"`
	}
	type Wrapper struct {
		Payload interface{} `json:"payload,omitempty"`
	}

	should := require.New(t)

	var payload *Payload
	wrapper := &Wrapper{
		Payload: payload,
	}

	err := json.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.NoError(err)
	should.NotNil(wrapper.Payload)
	should.Nil(payload)

	err = json.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.NoError(err)
	should.Nil(wrapper.Payload)
	should.Nil(payload)

	payload = nil
	wrapper = &Wrapper{
		Payload: payload,
	}

	err = jsoniter.Unmarshal([]byte(`{"payload": {"val": 42}}`), &wrapper)
	should.NoError(err)
	should.NotNil(wrapper.Payload)
	should.Nil(payload)

	err = jsoniter.Unmarshal([]byte(`{"payload": null}`), &wrapper)
	should.NoError(err)
	should.Nil(wrapper.Payload)
	should.Nil(payload)
}
