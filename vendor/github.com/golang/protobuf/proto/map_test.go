package proto_test

import (
	"fmt"
	"testing"

	"github.com/golang/protobuf/proto"
	ppb "github.com/golang/protobuf/proto/proto3_proto"
)

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
