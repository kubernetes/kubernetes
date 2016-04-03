package unmarshalmerge

import (
	"github.com/gogo/protobuf/proto"
	math_rand "math/rand"
	"testing"
	"time"
)

func TestUnmarshalMerge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedBig(popr, true)
	if p.GetSub() == nil {
		p.Sub = &Sub{SubNumber: proto.Int64(12345)}
	}
	data, err := proto.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := &Sub{}
	b := &Big{
		Sub: s,
	}
	err = proto.UnmarshalMerge(data, b)
	if err != nil {
		t.Fatal(err)
	}
	if s.GetSubNumber() != p.GetSub().GetSubNumber() {
		t.Fatalf("should have unmarshaled subnumber into sub")
	}
}

func TestUnsafeUnmarshalMerge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedBigUnsafe(popr, true)
	if p.GetSub() == nil {
		p.Sub = &Sub{SubNumber: proto.Int64(12345)}
	}
	data, err := proto.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := &Sub{}
	b := &BigUnsafe{
		Sub: s,
	}
	err = proto.UnmarshalMerge(data, b)
	if err != nil {
		t.Fatal(err)
	}

	if s.GetSubNumber() != p.GetSub().GetSubNumber() {
		t.Fatalf("should have unmarshaled subnumber into sub")
	}
}

func TestInt64Merge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedIntMerge(popr, true)
	p2 := NewPopulatedIntMerge(popr, true)
	data, err := proto.Marshal(p2)
	if err != nil {
		t.Fatal(err)
	}
	if err := proto.UnmarshalMerge(data, p); err != nil {
		t.Fatal(err)
	}
	if !p.Equal(p2) {
		t.Fatalf("exptected %#v but got %#v", p2, p)
	}
}
