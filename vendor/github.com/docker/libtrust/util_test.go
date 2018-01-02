package libtrust

import (
	"encoding/pem"
	"reflect"
	"testing"
)

func TestAddPEMHeadersToKey(t *testing.T) {
	pk := &rsaPublicKey{nil, map[string]interface{}{}}
	blk := &pem.Block{Headers: map[string]string{"hosts": "localhost,127.0.0.1"}}
	addPEMHeadersToKey(blk, pk)

	val := pk.GetExtendedField("hosts")
	hosts, ok := val.([]string)
	if !ok {
		t.Fatalf("hosts type(%v), expected []string", reflect.TypeOf(val))
	}
	expected := []string{"localhost", "127.0.0.1"}
	if !reflect.DeepEqual(hosts, expected) {
		t.Errorf("hosts(%v), expected %v", hosts, expected)
	}
}

func TestBase64URL(t *testing.T) {
	clean := "eyJhbGciOiJQQkVTMi1IUzI1NitBMTI4S1ciLCJwMnMiOiIyV0NUY0paMVJ2ZF9DSnVKcmlwUTF3IiwicDJjIjo0MDk2LCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiY3R5IjoiandrK2pzb24ifQ"

	tests := []string{
		clean, // clean roundtrip
		"eyJhbGciOiJQQkVTMi1IUzI1NitBMTI4S1ciLCJwMnMiOiIyV0NUY0paMVJ2\nZF9DSnVKcmlwUTF3IiwicDJjIjo0MDk2LCJlbmMiOiJBMTI4Q0JDLUhTMjU2\nIiwiY3R5IjoiandrK2pzb24ifQ",     // with newlines
		"eyJhbGciOiJQQkVTMi1IUzI1NitBMTI4S1ciLCJwMnMiOiIyV0NUY0paMVJ2 \n ZF9DSnVKcmlwUTF3IiwicDJjIjo0MDk2LCJlbmMiOiJBMTI4Q0JDLUhTMjU2 \n IiwiY3R5IjoiandrK2pzb24ifQ", // with newlines and spaces
	}

	for i, test := range tests {
		b, err := joseBase64UrlDecode(test)
		if err != nil {
			t.Fatalf("on test %d: %s", i, err)
		}
		got := joseBase64UrlEncode(b)

		if got != clean {
			t.Errorf("expected %q, got %q", clean, got)
		}
	}
}
