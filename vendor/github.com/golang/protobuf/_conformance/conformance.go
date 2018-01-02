// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
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

// conformance implements the conformance test subprocess protocol as
// documented in conformance.proto.
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	pb "github.com/golang/protobuf/_conformance/conformance_proto"
	"github.com/golang/protobuf/jsonpb"
	"github.com/golang/protobuf/proto"
)

func main() {
	var sizeBuf [4]byte
	inbuf := make([]byte, 0, 4096)
	outbuf := proto.NewBuffer(nil)
	for {
		if _, err := io.ReadFull(os.Stdin, sizeBuf[:]); err == io.EOF {
			break
		} else if err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: read request:", err)
			os.Exit(1)
		}
		size := binary.LittleEndian.Uint32(sizeBuf[:])
		if int(size) > cap(inbuf) {
			inbuf = make([]byte, size)
		}
		inbuf = inbuf[:size]
		if _, err := io.ReadFull(os.Stdin, inbuf); err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: read request:", err)
			os.Exit(1)
		}

		req := new(pb.ConformanceRequest)
		if err := proto.Unmarshal(inbuf, req); err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: parse request:", err)
			os.Exit(1)
		}
		res := handle(req)

		if err := outbuf.Marshal(res); err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: marshal response:", err)
			os.Exit(1)
		}
		binary.LittleEndian.PutUint32(sizeBuf[:], uint32(len(outbuf.Bytes())))
		if _, err := os.Stdout.Write(sizeBuf[:]); err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: write response:", err)
			os.Exit(1)
		}
		if _, err := os.Stdout.Write(outbuf.Bytes()); err != nil {
			fmt.Fprintln(os.Stderr, "go conformance: write response:", err)
			os.Exit(1)
		}
		outbuf.Reset()
	}
}

var jsonMarshaler = jsonpb.Marshaler{
	OrigName: true,
}

func handle(req *pb.ConformanceRequest) *pb.ConformanceResponse {
	var err error
	var msg pb.TestAllTypes
	switch p := req.Payload.(type) {
	case *pb.ConformanceRequest_ProtobufPayload:
		err = proto.Unmarshal(p.ProtobufPayload, &msg)
	case *pb.ConformanceRequest_JsonPayload:
		err = jsonpb.UnmarshalString(p.JsonPayload, &msg)
		if err != nil && err.Error() == "unmarshaling Any not supported yet" {
			return &pb.ConformanceResponse{
				Result: &pb.ConformanceResponse_Skipped{
					Skipped: err.Error(),
				},
			}
		}
	default:
		return &pb.ConformanceResponse{
			Result: &pb.ConformanceResponse_RuntimeError{
				RuntimeError: "unknown request payload type",
			},
		}
	}
	if err != nil {
		return &pb.ConformanceResponse{
			Result: &pb.ConformanceResponse_ParseError{
				ParseError: err.Error(),
			},
		}
	}
	switch req.RequestedOutputFormat {
	case pb.WireFormat_PROTOBUF:
		p, err := proto.Marshal(&msg)
		if err != nil {
			return &pb.ConformanceResponse{
				Result: &pb.ConformanceResponse_SerializeError{
					SerializeError: err.Error(),
				},
			}
		}
		return &pb.ConformanceResponse{
			Result: &pb.ConformanceResponse_ProtobufPayload{
				ProtobufPayload: p,
			},
		}
	case pb.WireFormat_JSON:
		p, err := jsonMarshaler.MarshalToString(&msg)
		if err != nil {
			return &pb.ConformanceResponse{
				Result: &pb.ConformanceResponse_SerializeError{
					SerializeError: err.Error(),
				},
			}
		}
		return &pb.ConformanceResponse{
			Result: &pb.ConformanceResponse_JsonPayload{
				JsonPayload: p,
			},
		}
	default:
		return &pb.ConformanceResponse{
			Result: &pb.ConformanceResponse_RuntimeError{
				RuntimeError: "unknown output format",
			},
		}
	}
}
