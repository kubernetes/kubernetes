package test

import (
	"github.com/gogo/protobuf/proto"
	"testing"
)

func TestBugUuid(t *testing.T) {
	u := &CustomContainer{CustomStruct: NidOptCustom{Id: Uuid{}}}
	data, err := proto.Marshal(u)
	if err != nil {
		panic(err)
	}
	u2 := &CustomContainer{}
	err = proto.Unmarshal(data, u2)
	if err != nil {
		panic(err)
	}
	t.Logf("%+v", u2)
	if u2.CustomStruct.Id != nil {
		t.Fatalf("should be nil")
	}
}
