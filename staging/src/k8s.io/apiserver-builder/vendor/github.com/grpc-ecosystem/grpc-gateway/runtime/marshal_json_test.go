package runtime_test

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes/empty"
	structpb "github.com/golang/protobuf/ptypes/struct"
	"github.com/golang/protobuf/ptypes/timestamp"
	"github.com/golang/protobuf/ptypes/wrappers"
	"github.com/grpc-ecosystem/grpc-gateway/examples/examplepb"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
)

func TestJSONBuiltinMarshal(t *testing.T) {
	var m runtime.JSONBuiltin
	msg := examplepb.SimpleMessage{
		Id: "foo",
	}

	buf, err := m.Marshal(&msg)
	if err != nil {
		t.Errorf("m.Marshal(%v) failed with %v; want success", &msg, err)
	}

	var got examplepb.SimpleMessage
	if err := json.Unmarshal(buf, &got); err != nil {
		t.Errorf("json.Unmarshal(%q, &got) failed with %v; want success", buf, err)
	}
	if want := msg; !reflect.DeepEqual(got, want) {
		t.Errorf("got = %v; want %v", &got, &want)
	}
}

func TestJSONBuiltinMarshalField(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinFieldFixtures {
		buf, err := m.Marshal(fixt.data)
		if err != nil {
			t.Errorf("m.Marshal(%v) failed with %v; want success", fixt.data, err)
		}
		if got, want := string(buf), fixt.json; got != want {
			t.Errorf("got = %q; want %q; data = %#v", got, want, fixt.data)
		}
	}
}

func TestJSONBuiltinMarshalFieldKnownErrors(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinKnownErrors {
		buf, err := m.Marshal(fixt.data)
		if err != nil {
			t.Errorf("m.Marshal(%v) failed with %v; want success", fixt.data, err)
		}
		if got, want := string(buf), fixt.json; got == want {
			t.Errorf("surprisingly got = %q; as want %q; data = %#v", got, want, fixt.data)
		}
	}
}

func TestJSONBuiltinsnmarshal(t *testing.T) {
	var (
		m   runtime.JSONBuiltin
		got examplepb.SimpleMessage

		data = []byte(`{"id": "foo"}`)
	)
	if err := m.Unmarshal(data, &got); err != nil {
		t.Errorf("m.Unmarshal(%q, &got) failed with %v; want success", data, err)
	}

	want := examplepb.SimpleMessage{
		Id: "foo",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got = %v; want = %v", &got, &want)
	}
}

func TestJSONBuiltinUnmarshalField(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinFieldFixtures {
		dest := reflect.New(reflect.TypeOf(fixt.data))
		if err := m.Unmarshal([]byte(fixt.json), dest.Interface()); err != nil {
			t.Errorf("m.Unmarshal(%q, dest) failed with %v; want success", fixt.json, err)
		}

		if got, want := dest.Elem().Interface(), fixt.data; !reflect.DeepEqual(got, want) {
			t.Errorf("got = %#v; want = %#v; input = %q", got, want, fixt.json)
		}
	}
}

func TestJSONBuiltinUnmarshalFieldKnownErrors(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinKnownErrors {
		dest := reflect.New(reflect.TypeOf(fixt.data))
		if err := m.Unmarshal([]byte(fixt.json), dest.Interface()); err == nil {
			t.Errorf("m.Unmarshal(%q, dest) succeeded; want ane error", fixt.json)
		}
	}
}

func TestJSONBuiltinEncoder(t *testing.T) {
	var m runtime.JSONBuiltin
	msg := examplepb.SimpleMessage{
		Id: "foo",
	}

	var buf bytes.Buffer
	enc := m.NewEncoder(&buf)
	if err := enc.Encode(&msg); err != nil {
		t.Errorf("enc.Encode(%v) failed with %v; want success", &msg, err)
	}

	var got examplepb.SimpleMessage
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Errorf("json.Unmarshal(%q, &got) failed with %v; want success", buf.String(), err)
	}
	if want := msg; !reflect.DeepEqual(got, want) {
		t.Errorf("got = %v; want %v", &got, &want)
	}
}

func TestJSONBuiltinEncoderFields(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinFieldFixtures {
		var buf bytes.Buffer
		enc := m.NewEncoder(&buf)
		if err := enc.Encode(fixt.data); err != nil {
			t.Errorf("enc.Encode(%#v) failed with %v; want success", fixt.data, err)
		}

		if got, want := buf.String(), fixt.json+"\n"; got != want {
			t.Errorf("got = %q; want %q; data = %#v", got, want, fixt.data)
		}
	}
}

func TestJSONBuiltinDecoder(t *testing.T) {
	var (
		m   runtime.JSONBuiltin
		got examplepb.SimpleMessage

		data = `{"id": "foo"}`
	)
	r := strings.NewReader(data)
	dec := m.NewDecoder(r)
	if err := dec.Decode(&got); err != nil {
		t.Errorf("m.Unmarshal(&got) failed with %v; want success", err)
	}

	want := examplepb.SimpleMessage{
		Id: "foo",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got = %v; want = %v", &got, &want)
	}
}

func TestJSONBuiltinDecoderFields(t *testing.T) {
	var m runtime.JSONBuiltin
	for _, fixt := range builtinFieldFixtures {
		r := strings.NewReader(fixt.json)
		dec := m.NewDecoder(r)
		dest := reflect.New(reflect.TypeOf(fixt.data))
		if err := dec.Decode(dest.Interface()); err != nil {
			t.Errorf("dec.Decode(dest) failed with %v; want success; data = %q", err, fixt.json)
		}

		if got, want := dest.Elem().Interface(), fixt.data; !reflect.DeepEqual(got, want) {
			t.Errorf("got = %v; want = %v; input = %q", got, want, fixt.json)
		}
	}
}

var (
	builtinFieldFixtures = []struct {
		data interface{}
		json string
	}{
		{data: "", json: `""`},
		{data: proto.String(""), json: `""`},
		{data: "foo", json: `"foo"`},
		{data: proto.String("foo"), json: `"foo"`},
		{data: int32(-1), json: "-1"},
		{data: proto.Int32(-1), json: "-1"},
		{data: int64(-1), json: "-1"},
		{data: proto.Int64(-1), json: "-1"},
		{data: uint32(123), json: "123"},
		{data: proto.Uint32(123), json: "123"},
		{data: uint64(123), json: "123"},
		{data: proto.Uint64(123), json: "123"},
		{data: float32(-1.5), json: "-1.5"},
		{data: proto.Float32(-1.5), json: "-1.5"},
		{data: float64(-1.5), json: "-1.5"},
		{data: proto.Float64(-1.5), json: "-1.5"},
		{data: true, json: "true"},
		{data: proto.Bool(true), json: "true"},
		{data: (*string)(nil), json: "null"},
		{data: new(empty.Empty), json: "{}"},
		{data: examplepb.NumericEnum_ONE, json: "1"},
		{
			data: (*examplepb.NumericEnum)(proto.Int32(int32(examplepb.NumericEnum_ONE))),
			json: "1",
		},
	}
	builtinKnownErrors = []struct {
		data interface{}
		json string
	}{
		{data: examplepb.NumericEnum_ONE, json: "ONE"},
		{
			data: (*examplepb.NumericEnum)(proto.Int32(int32(examplepb.NumericEnum_ONE))),
			json: "ONE",
		},
		{
			data: &examplepb.ABitOfEverything_OneofString{OneofString: "abc"},
			json: `"abc"`,
		},
		{
			data: &timestamp.Timestamp{
				Seconds: 1462875553,
				Nanos:   123000000,
			},
			json: `"2016-05-10T10:19:13.123Z"`,
		},
		{
			data: &wrappers.Int32Value{Value: 123},
			json: "123",
		},
		{
			data: &structpb.Value{
				Kind: &structpb.Value_StringValue{
					StringValue: "abc",
				},
			},
			json: `"abc"`,
		},
	}
)
