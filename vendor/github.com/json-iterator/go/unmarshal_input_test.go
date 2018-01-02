package jsoniter

import (
	"encoding/json"
	"reflect"
	"testing"

	fuzz "github.com/google/gofuzz"
)

func Test_NilInput(t *testing.T) {
	var jb []byte // nil
	var out string
	err := Unmarshal(jb, &out)
	if err == nil {
		t.Errorf("Expected error")
	}
}

func Test_EmptyInput(t *testing.T) {
	jb := []byte("")
	var out string
	err := Unmarshal(jb, &out)
	if err == nil {
		t.Errorf("Expected error")
	}
}

func Test_RandomInput_Bytes(t *testing.T) {
	fz := fuzz.New().NilChance(0)
	for i := 0; i < 10000; i++ {
		var jb []byte
		fz.Fuzz(&jb)
		testRandomInput(t, jb)
	}
}

func Test_RandomInput_String(t *testing.T) {
	fz := fuzz.New().NilChance(0)
	for i := 0; i < 10000; i++ {
		var js string
		fz.Fuzz(&js)
		jb := []byte(js)
		testRandomInput(t, jb)
	}
}

func testRandomInput(t *testing.T, jb []byte) {
	var outString string
	testRandomInputTo(t, jb, &outString)

	var outInt int
	testRandomInputTo(t, jb, &outInt)

	var outStruct struct{}
	testRandomInputTo(t, jb, &outStruct)

	var outSlice []string
	testRandomInputTo(t, jb, &outSlice)
}

func testRandomInputTo(t *testing.T, jb []byte, out interface{}) {
	err := Unmarshal(jb, out)
	if err == nil {
		// Cross-check stdlib to see if we just happened to fuzz a legit value.
		err := json.Unmarshal(jb, out)
		if err != nil {
			t.Fatalf("Expected error unmarshaling as %s:\nas string: %q\nas bytes: %v",
				reflect.TypeOf(out).Elem().Kind(), string(jb), jb)
		}
	}
}
