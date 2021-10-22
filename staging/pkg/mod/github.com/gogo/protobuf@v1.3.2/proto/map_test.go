package proto_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/gogo/protobuf/proto"
	ppb "github.com/gogo/protobuf/proto/proto3_proto"
)

func TestMap(t *testing.T) {
	var b []byte
	fmt.Sscanf("a2010c0a044b657931120456616c31a201130a044b657932120556616c3261120456616c32a201240a044b6579330d05000000120556616c33621a0556616c3361120456616c331505000000a20100a201260a044b657934130a07536f6d6555524c1209536f6d655469746c651a08536e69707065743114", "%x", &b)

	var m ppb.Message
	if err := proto.Unmarshal(b, &m); err != nil {
		t.Fatalf("proto.Unmarshal error: %v", err)
	}

	got := m.StringMap
	want := map[string]string{
		"":     "",
		"Key1": "Val1",
		"Key2": "Val2",
		"Key3": "Val3",
		"Key4": "",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("maps differ:\ngot  %#v\nwant %#v", got, want)
	}
}

func marshalled() []byte {
	m := &ppb.IntMaps{}
	for i := 0; i < 1000; i++ {
		m.Maps = append(m.Maps, &ppb.IntMap{
			Rtt: map[int32]int32{1: 2},
		})
	}
	b, err := proto.Marshal(m)
	if err != nil {
		panic(fmt.Sprintf("Can't marshal %+v: %v", m, err))
	}
	return b
}

func BenchmarkConcurrentMapUnmarshal(b *testing.B) {
	in := marshalled()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var out ppb.IntMaps
			if err := proto.Unmarshal(in, &out); err != nil {
				b.Errorf("Can't unmarshal ppb.IntMaps: %v", err)
			}
		}
	})
}

func BenchmarkSequentialMapUnmarshal(b *testing.B) {
	in := marshalled()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out ppb.IntMaps
		if err := proto.Unmarshal(in, &out); err != nil {
			b.Errorf("Can't unmarshal ppb.IntMaps: %v", err)
		}
	}
}
