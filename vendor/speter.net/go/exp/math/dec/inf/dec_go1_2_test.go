// +build go1.2

package inf

import (
	"encoding"
	"encoding/json"
	"testing"
)

var _ encoding.TextMarshaler = new(Dec)
var _ encoding.TextUnmarshaler = new(Dec)

type Obj struct {
	Val *Dec
}

func TestDecJsonMarshalUnmarshal(t *testing.T) {
	o := Obj{Val: NewDec(123, 2)}
	js, err := json.Marshal(o)
	if err != nil {
		t.Fatalf("json.Marshal(%v): got %v, want ok", o, err)
	}
	o2 := &Obj{}
	err = json.Unmarshal(js, o2)
	if err != nil {
		t.Fatalf("json.Unmarshal(%#q): got %v, want ok", js, err)
	}
	if o.Val.Scale() != o2.Val.Scale() ||
		o.Val.UnscaledBig().Cmp(o2.Val.UnscaledBig()) != 0 {
		t.Fatalf("json.Unmarshal(json.Marshal(%v)): want %v, got %v", o, o, o2)
	}
}
