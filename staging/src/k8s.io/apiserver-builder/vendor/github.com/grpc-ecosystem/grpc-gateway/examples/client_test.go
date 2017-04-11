package main

import (
	"reflect"
	"testing"

	"github.com/grpc-ecosystem/grpc-gateway/examples/clients/abe"
	"github.com/grpc-ecosystem/grpc-gateway/examples/clients/echo"
)

func TestClientIntegration(t *testing.T) {
}

func TestEchoClient(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}

	cl := echo.NewEchoServiceApiWithBasePath("http://localhost:8080")
	resp, err := cl.Echo("foo")
	if err != nil {
		t.Errorf(`cl.Echo("foo") failed with %v; want success`, err)
	}
	if got, want := resp.Id, "foo"; got != want {
		t.Errorf("resp.Id = %q; want %q", got, want)
	}
}

func TestEchoBodyClient(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}

	cl := echo.NewEchoServiceApiWithBasePath("http://localhost:8080")
	req := echo.ExamplepbSimpleMessage{Id: "foo"}
	resp, err := cl.EchoBody(req)
	if err != nil {
		t.Errorf("cl.EchoBody(%#v) failed with %v; want success", req, err)
	}
	if got, want := resp.Id, "foo"; got != want {
		t.Errorf("resp.Id = %q; want %q", got, want)
	}
}

func TestAbitOfEverythingClient(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}

	cl := abe.NewABitOfEverythingServiceApiWithBasePath("http://localhost:8080")
	testABEClientCreate(t, cl)
	testABEClientCreateBody(t, cl)
}

func testABEClientCreate(t *testing.T, cl *abe.ABitOfEverythingServiceApi) {
	want := abe.ExamplepbABitOfEverything{
		FloatValue:               1.5,
		DoubleValue:              2.5,
		Int64Value:               "4294967296",
		Uint64Value:              "9223372036854775807",
		Int32Value:               -2147483648,
		Fixed64Value:             "9223372036854775807",
		Fixed32Value:             4294967295,
		BoolValue:                true,
		StringValue:              "strprefix/foo",
		Uint32Value:              4294967295,
		Sfixed32Value:            2147483647,
		Sfixed64Value:            "-4611686018427387904",
		Sint32Value:              2147483647,
		Sint64Value:              "4611686018427387903",
		NonConventionalNameValue: "camelCase",
	}
	resp, err := cl.Create(
		want.FloatValue,
		want.DoubleValue,
		want.Int64Value,
		want.Uint64Value,
		want.Int32Value,
		want.Fixed64Value,
		want.Fixed32Value,
		want.BoolValue,
		want.StringValue,
		want.Uint32Value,
		want.Sfixed32Value,
		want.Sfixed64Value,
		want.Sint32Value,
		want.Sint64Value,
		want.NonConventionalNameValue,
	)
	if err != nil {
		t.Errorf("cl.Create(%#v) failed with %v; want success", want, err)
	}
	if resp.Uuid == "" {
		t.Errorf("resp.Uuid is empty; want not empty")
	}
	resp.Uuid = ""
	if got := resp; !reflect.DeepEqual(got, want) {
		t.Errorf("resp = %#v; want %#v", got, want)
	}
}

func testABEClientCreateBody(t *testing.T, cl *abe.ABitOfEverythingServiceApi) {
	t.Log("TODO: support enum")
	return

	want := abe.ExamplepbABitOfEverything{
		FloatValue:               1.5,
		DoubleValue:              2.5,
		Int64Value:               "4294967296",
		Uint64Value:              "9223372036854775807",
		Int32Value:               -2147483648,
		Fixed64Value:             "9223372036854775807",
		Fixed32Value:             4294967295,
		BoolValue:                true,
		StringValue:              "strprefix/foo",
		Uint32Value:              4294967295,
		Sfixed32Value:            2147483647,
		Sfixed64Value:            "-4611686018427387904",
		Sint32Value:              2147483647,
		Sint64Value:              "4611686018427387903",
		NonConventionalNameValue: "camelCase",

		Nested: []abe.ABitOfEverythingNested{
			{
				Name:   "bar",
				Amount: 10,
			},
			{
				Name:   "baz",
				Amount: 20,
			},
		},
		RepeatedStringValue: []string{"a", "b", "c"},
		OneofString:         "x",
		MapValue:            map[string]abe.ExamplepbNumericEnum{
		// "a": abe.ExamplepbNumericEnum_ONE,
		// "b": abe.ExamplepbNumericEnum_ZERO,
		},
		MappedStringValue: map[string]string{
			"a": "x",
			"b": "y",
		},
		MappedNestedValue: map[string]abe.ABitOfEverythingNested{
			"a": {Name: "x", Amount: 1},
			"b": {Name: "y", Amount: 2},
		},
	}
	resp, err := cl.CreateBody(want)
	if err != nil {
		t.Errorf("cl.CreateBody(%#v) failed with %v; want success", want, err)
	}
	if resp.Uuid == "" {
		t.Errorf("resp.Uuid is empty; want not empty")
	}
	resp.Uuid = ""
	if got := resp; !reflect.DeepEqual(got, want) {
		t.Errorf("resp = %#v; want %#v", got, want)
	}
}
