// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto_test

import (
	"testing"

	"github.com/gogo/protobuf/proto"
	tpb "github.com/gogo/protobuf/proto/proto3_proto"
)

var (
	bytesBlackhole []byte
	msgBlackhole   = new(tpb.Message)
)

// Disabled this Benchmark because it is using features (b.Run) from go1.7 and gogoprotobuf still have compatibility with go1.5
// BenchmarkVarint32ArraySmall shows the performance on an array of small int32 fields (1 and
// 2 bytes long).
// func BenchmarkVarint32ArraySmall(b *testing.B) {
// 	for i := uint(1); i <= 10; i++ {
// 		dist := genInt32Dist([7]int{0, 3, 1}, 1<<i)
// 		raw, err := proto.Marshal(&tpb.Message{
// 			ShortKey: dist,
// 		})
// 		if err != nil {
// 			b.Error("wrong encode", err)
// 		}
// 		b.Run(fmt.Sprintf("Len%v", len(dist)), func(b *testing.B) {
// 			scratchBuf := proto.NewBuffer(nil)
// 			b.ResetTimer()
// 			for k := 0; k < b.N; k++ {
// 				scratchBuf.SetBuf(raw)
// 				msgBlackhole.Reset()
// 				if err := scratchBuf.Unmarshal(msgBlackhole); err != nil {
// 					b.Error("wrong decode", err)
// 				}
// 			}
// 		})
// 	}
// }

// Disabled this Benchmark because it is using features (b.Run) from go1.7 and gogoprotobuf still have compatibility with go1.5
// BenchmarkVarint32ArrayLarge shows the performance on an array of large int32 fields (3 and
// 4 bytes long, with a small number of 1, 2, 5 and 10 byte long versions).
// func BenchmarkVarint32ArrayLarge(b *testing.B) {
// 	for i := uint(1); i <= 10; i++ {
// 		dist := genInt32Dist([7]int{0, 1, 2, 4, 8, 1, 1}, 1<<i)
// 		raw, err := proto.Marshal(&tpb.Message{
// 			ShortKey: dist,
// 		})
// 		if err != nil {
// 			b.Error("wrong encode", err)
// 		}
// 		b.Run(fmt.Sprintf("Len%v", len(dist)), func(b *testing.B) {
// 			scratchBuf := proto.NewBuffer(nil)
// 			b.ResetTimer()
// 			for k := 0; k < b.N; k++ {
// 				scratchBuf.SetBuf(raw)
// 				msgBlackhole.Reset()
// 				if err := scratchBuf.Unmarshal(msgBlackhole); err != nil {
// 					b.Error("wrong decode", err)
// 				}
// 			}
// 		})
// 	}
// }

// Disabled this Benchmark because it is using features (b.Run) from go1.7 and gogoprotobuf still have compatibility with go1.5
// BenchmarkVarint64ArraySmall shows the performance on an array of small int64 fields (1 and
// 2 bytes long).
// func BenchmarkVarint64ArraySmall(b *testing.B) {
// 	for i := uint(1); i <= 10; i++ {
// 		dist := genUint64Dist([11]int{0, 3, 1}, 1<<i)
// 		raw, err := proto.Marshal(&tpb.Message{
// 			Key: dist,
// 		})
// 		if err != nil {
// 			b.Error("wrong encode", err)
// 		}
// 		b.Run(fmt.Sprintf("Len%v", len(dist)), func(b *testing.B) {
// 			scratchBuf := proto.NewBuffer(nil)
// 			b.ResetTimer()
// 			for k := 0; k < b.N; k++ {
// 				scratchBuf.SetBuf(raw)
// 				msgBlackhole.Reset()
// 				if err := scratchBuf.Unmarshal(msgBlackhole); err != nil {
// 					b.Error("wrong decode", err)
// 				}
// 			}
// 		})
// 	}
// }

// Disabled this Benchmark because it is using features (b.Run) from go1.7 and gogoprotobuf still have compatibility with go1.5
// BenchmarkVarint64ArrayLarge shows the performance on an array of large int64 fields (6, 7,
// and 8 bytes long with a small number of the other sizes).
// func BenchmarkVarint64ArrayLarge(b *testing.B) {
// 	for i := uint(1); i <= 10; i++ {
// 		dist := genUint64Dist([11]int{0, 1, 1, 2, 4, 8, 16, 32, 16, 1, 1}, 1<<i)
// 		raw, err := proto.Marshal(&tpb.Message{
// 			Key: dist,
// 		})
// 		if err != nil {
// 			b.Error("wrong encode", err)
// 		}
// 		b.Run(fmt.Sprintf("Len%v", len(dist)), func(b *testing.B) {
// 			scratchBuf := proto.NewBuffer(nil)
// 			b.ResetTimer()
// 			for k := 0; k < b.N; k++ {
// 				scratchBuf.SetBuf(raw)
// 				msgBlackhole.Reset()
// 				if err := scratchBuf.Unmarshal(msgBlackhole); err != nil {
// 					b.Error("wrong decode", err)
// 				}
// 			}
// 		})
// 	}
// }

// Disabled this Benchmark because it is using features (b.Run) from go1.7 and gogoprotobuf still have compatibility with go1.5
// BenchmarkVarint64ArrayMixed shows the performance of lots of small messages, each
// containing a small number of large (3, 4, and 5 byte) repeated int64s.
// func BenchmarkVarint64ArrayMixed(b *testing.B) {
// 	for i := uint(1); i <= 1<<5; i <<= 1 {
// 		dist := genUint64Dist([11]int{0, 0, 0, 4, 6, 4, 0, 0, 0, 0, 0}, int(i))
// 		// number of sub fields
// 		for k := uint(1); k <= 1<<10; k <<= 2 {
// 			msg := &tpb.Message{}
// 			for m := uint(0); m < k; m++ {
// 				msg.Children = append(msg.Children, &tpb.Message{
// 					Key: dist,
// 				})
// 			}
// 			raw, err := proto.Marshal(msg)
// 			if err != nil {
// 				b.Error("wrong encode", err)
// 			}
// 			b.Run(fmt.Sprintf("Fields%vLen%v", k, i), func(b *testing.B) {
// 				scratchBuf := proto.NewBuffer(nil)
// 				b.ResetTimer()
// 				for k := 0; k < b.N; k++ {
// 					scratchBuf.SetBuf(raw)
// 					msgBlackhole.Reset()
// 					if err := scratchBuf.Unmarshal(msgBlackhole); err != nil {
// 						b.Error("wrong decode", err)
// 					}
// 				}
// 			})
// 		}
// 	}
// }

// genInt32Dist generates a slice of ints that will match the size distribution of dist.
// A size of 6 corresponds to a max length varint32, which is 10 bytes.  The distribution
// is 1-indexed. (i.e. the value at index 1 is how many 1 byte ints to create).
func genInt32Dist(dist [7]int, count int) (dest []int32) {
	for i := 0; i < count; i++ {
		for k := 0; k < len(dist); k++ {
			var num int32
			switch k {
			case 1:
				num = 1<<7 - 1
			case 2:
				num = 1<<14 - 1
			case 3:
				num = 1<<21 - 1
			case 4:
				num = 1<<28 - 1
			case 5:
				num = 1<<29 - 1
			case 6:
				num = -1
			}
			for m := 0; m < dist[k]; m++ {
				dest = append(dest, num)
			}
		}
	}
	return
}

// genUint64Dist generates a slice of ints that will match the size distribution of dist.
// The distribution is 1-indexed. (i.e. the value at index 1 is how many 1 byte ints to create).
func genUint64Dist(dist [11]int, count int) (dest []uint64) {
	for i := 0; i < count; i++ {
		for k := 0; k < len(dist); k++ {
			var num uint64
			switch k {
			case 1:
				num = 1<<7 - 1
			case 2:
				num = 1<<14 - 1
			case 3:
				num = 1<<21 - 1
			case 4:
				num = 1<<28 - 1
			case 5:
				num = 1<<35 - 1
			case 6:
				num = 1<<42 - 1
			case 7:
				num = 1<<49 - 1
			case 8:
				num = 1<<56 - 1
			case 9:
				num = 1<<63 - 1
			case 10:
				num = 1<<64 - 1
			}
			for m := 0; m < dist[k]; m++ {
				dest = append(dest, num)
			}
		}
	}
	return
}

// BenchmarkDecodeEmpty measures the overhead of doing the minimal possible decode.
func BenchmarkDecodeEmpty(b *testing.B) {
	raw, err := proto.Marshal(&tpb.Message{})
	if err != nil {
		b.Error("wrong encode", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := proto.Unmarshal(raw, msgBlackhole); err != nil {
			b.Error("wrong decode", err)
		}
	}
}
