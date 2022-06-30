/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package binarylog

import (
	"net"
	"strings"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	pb "google.golang.org/grpc/binarylog/grpc_binarylog_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type callIDGenerator struct {
	id uint64
}

func (g *callIDGenerator) next() uint64 {
	id := atomic.AddUint64(&g.id, 1)
	return id
}

// reset is for testing only, and doesn't need to be thread safe.
func (g *callIDGenerator) reset() {
	g.id = 0
}

var idGen callIDGenerator

// MethodLogger is the sub-logger for each method.
type MethodLogger interface {
	Log(LogEntryConfig)
}

type methodLogger struct {
	headerMaxLen, messageMaxLen uint64

	callID          uint64
	idWithinCallGen *callIDGenerator

	sink Sink // TODO(blog): make this plugable.
}

func newMethodLogger(h, m uint64) *methodLogger {
	return &methodLogger{
		headerMaxLen:  h,
		messageMaxLen: m,

		callID:          idGen.next(),
		idWithinCallGen: &callIDGenerator{},

		sink: DefaultSink, // TODO(blog): make it plugable.
	}
}

// Build is an internal only method for building the proto message out of the
// input event. It's made public to enable other library to reuse as much logic
// in methodLogger as possible.
func (ml *methodLogger) Build(c LogEntryConfig) *pb.GrpcLogEntry {
	m := c.toProto()
	timestamp, _ := ptypes.TimestampProto(time.Now())
	m.Timestamp = timestamp
	m.CallId = ml.callID
	m.SequenceIdWithinCall = ml.idWithinCallGen.next()

	switch pay := m.Payload.(type) {
	case *pb.GrpcLogEntry_ClientHeader:
		m.PayloadTruncated = ml.truncateMetadata(pay.ClientHeader.GetMetadata())
	case *pb.GrpcLogEntry_ServerHeader:
		m.PayloadTruncated = ml.truncateMetadata(pay.ServerHeader.GetMetadata())
	case *pb.GrpcLogEntry_Message:
		m.PayloadTruncated = ml.truncateMessage(pay.Message)
	}
	return m
}

// Log creates a proto binary log entry, and logs it to the sink.
func (ml *methodLogger) Log(c LogEntryConfig) {
	ml.sink.Write(ml.Build(c))
}

func (ml *methodLogger) truncateMetadata(mdPb *pb.Metadata) (truncated bool) {
	if ml.headerMaxLen == maxUInt {
		return false
	}
	var (
		bytesLimit = ml.headerMaxLen
		index      int
	)
	// At the end of the loop, index will be the first entry where the total
	// size is greater than the limit:
	//
	// len(entry[:index]) <= ml.hdr && len(entry[:index+1]) > ml.hdr.
	for ; index < len(mdPb.Entry); index++ {
		entry := mdPb.Entry[index]
		if entry.Key == "grpc-trace-bin" {
			// "grpc-trace-bin" is a special key. It's kept in the log entry,
			// but not counted towards the size limit.
			continue
		}
		currentEntryLen := uint64(len(entry.Value))
		if currentEntryLen > bytesLimit {
			break
		}
		bytesLimit -= currentEntryLen
	}
	truncated = index < len(mdPb.Entry)
	mdPb.Entry = mdPb.Entry[:index]
	return truncated
}

func (ml *methodLogger) truncateMessage(msgPb *pb.Message) (truncated bool) {
	if ml.messageMaxLen == maxUInt {
		return false
	}
	if ml.messageMaxLen >= uint64(len(msgPb.Data)) {
		return false
	}
	msgPb.Data = msgPb.Data[:ml.messageMaxLen]
	return true
}

// LogEntryConfig represents the configuration for binary log entry.
type LogEntryConfig interface {
	toProto() *pb.GrpcLogEntry
}

// ClientHeader configs the binary log entry to be a ClientHeader entry.
type ClientHeader struct {
	OnClientSide bool
	Header       metadata.MD
	MethodName   string
	Authority    string
	Timeout      time.Duration
	// PeerAddr is required only when it's on server side.
	PeerAddr net.Addr
}

func (c *ClientHeader) toProto() *pb.GrpcLogEntry {
	// This function doesn't need to set all the fields (e.g. seq ID). The Log
	// function will set the fields when necessary.
	clientHeader := &pb.ClientHeader{
		Metadata:   mdToMetadataProto(c.Header),
		MethodName: c.MethodName,
		Authority:  c.Authority,
	}
	if c.Timeout > 0 {
		clientHeader.Timeout = ptypes.DurationProto(c.Timeout)
	}
	ret := &pb.GrpcLogEntry{
		Type: pb.GrpcLogEntry_EVENT_TYPE_CLIENT_HEADER,
		Payload: &pb.GrpcLogEntry_ClientHeader{
			ClientHeader: clientHeader,
		},
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	if c.PeerAddr != nil {
		ret.Peer = addrToProto(c.PeerAddr)
	}
	return ret
}

// ServerHeader configs the binary log entry to be a ServerHeader entry.
type ServerHeader struct {
	OnClientSide bool
	Header       metadata.MD
	// PeerAddr is required only when it's on client side.
	PeerAddr net.Addr
}

func (c *ServerHeader) toProto() *pb.GrpcLogEntry {
	ret := &pb.GrpcLogEntry{
		Type: pb.GrpcLogEntry_EVENT_TYPE_SERVER_HEADER,
		Payload: &pb.GrpcLogEntry_ServerHeader{
			ServerHeader: &pb.ServerHeader{
				Metadata: mdToMetadataProto(c.Header),
			},
		},
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	if c.PeerAddr != nil {
		ret.Peer = addrToProto(c.PeerAddr)
	}
	return ret
}

// ClientMessage configs the binary log entry to be a ClientMessage entry.
type ClientMessage struct {
	OnClientSide bool
	// Message can be a proto.Message or []byte. Other messages formats are not
	// supported.
	Message interface{}
}

func (c *ClientMessage) toProto() *pb.GrpcLogEntry {
	var (
		data []byte
		err  error
	)
	if m, ok := c.Message.(proto.Message); ok {
		data, err = proto.Marshal(m)
		if err != nil {
			grpclogLogger.Infof("binarylogging: failed to marshal proto message: %v", err)
		}
	} else if b, ok := c.Message.([]byte); ok {
		data = b
	} else {
		grpclogLogger.Infof("binarylogging: message to log is neither proto.message nor []byte")
	}
	ret := &pb.GrpcLogEntry{
		Type: pb.GrpcLogEntry_EVENT_TYPE_CLIENT_MESSAGE,
		Payload: &pb.GrpcLogEntry_Message{
			Message: &pb.Message{
				Length: uint32(len(data)),
				Data:   data,
			},
		},
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	return ret
}

// ServerMessage configs the binary log entry to be a ServerMessage entry.
type ServerMessage struct {
	OnClientSide bool
	// Message can be a proto.Message or []byte. Other messages formats are not
	// supported.
	Message interface{}
}

func (c *ServerMessage) toProto() *pb.GrpcLogEntry {
	var (
		data []byte
		err  error
	)
	if m, ok := c.Message.(proto.Message); ok {
		data, err = proto.Marshal(m)
		if err != nil {
			grpclogLogger.Infof("binarylogging: failed to marshal proto message: %v", err)
		}
	} else if b, ok := c.Message.([]byte); ok {
		data = b
	} else {
		grpclogLogger.Infof("binarylogging: message to log is neither proto.message nor []byte")
	}
	ret := &pb.GrpcLogEntry{
		Type: pb.GrpcLogEntry_EVENT_TYPE_SERVER_MESSAGE,
		Payload: &pb.GrpcLogEntry_Message{
			Message: &pb.Message{
				Length: uint32(len(data)),
				Data:   data,
			},
		},
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	return ret
}

// ClientHalfClose configs the binary log entry to be a ClientHalfClose entry.
type ClientHalfClose struct {
	OnClientSide bool
}

func (c *ClientHalfClose) toProto() *pb.GrpcLogEntry {
	ret := &pb.GrpcLogEntry{
		Type:    pb.GrpcLogEntry_EVENT_TYPE_CLIENT_HALF_CLOSE,
		Payload: nil, // No payload here.
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	return ret
}

// ServerTrailer configs the binary log entry to be a ServerTrailer entry.
type ServerTrailer struct {
	OnClientSide bool
	Trailer      metadata.MD
	// Err is the status error.
	Err error
	// PeerAddr is required only when it's on client side and the RPC is trailer
	// only.
	PeerAddr net.Addr
}

func (c *ServerTrailer) toProto() *pb.GrpcLogEntry {
	st, ok := status.FromError(c.Err)
	if !ok {
		grpclogLogger.Info("binarylogging: error in trailer is not a status error")
	}
	var (
		detailsBytes []byte
		err          error
	)
	stProto := st.Proto()
	if stProto != nil && len(stProto.Details) != 0 {
		detailsBytes, err = proto.Marshal(stProto)
		if err != nil {
			grpclogLogger.Infof("binarylogging: failed to marshal status proto: %v", err)
		}
	}
	ret := &pb.GrpcLogEntry{
		Type: pb.GrpcLogEntry_EVENT_TYPE_SERVER_TRAILER,
		Payload: &pb.GrpcLogEntry_Trailer{
			Trailer: &pb.Trailer{
				Metadata:      mdToMetadataProto(c.Trailer),
				StatusCode:    uint32(st.Code()),
				StatusMessage: st.Message(),
				StatusDetails: detailsBytes,
			},
		},
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	if c.PeerAddr != nil {
		ret.Peer = addrToProto(c.PeerAddr)
	}
	return ret
}

// Cancel configs the binary log entry to be a Cancel entry.
type Cancel struct {
	OnClientSide bool
}

func (c *Cancel) toProto() *pb.GrpcLogEntry {
	ret := &pb.GrpcLogEntry{
		Type:    pb.GrpcLogEntry_EVENT_TYPE_CANCEL,
		Payload: nil,
	}
	if c.OnClientSide {
		ret.Logger = pb.GrpcLogEntry_LOGGER_CLIENT
	} else {
		ret.Logger = pb.GrpcLogEntry_LOGGER_SERVER
	}
	return ret
}

// metadataKeyOmit returns whether the metadata entry with this key should be
// omitted.
func metadataKeyOmit(key string) bool {
	switch key {
	case "lb-token", ":path", ":authority", "content-encoding", "content-type", "user-agent", "te":
		return true
	case "grpc-trace-bin": // grpc-trace-bin is special because it's visiable to users.
		return false
	}
	return strings.HasPrefix(key, "grpc-")
}

func mdToMetadataProto(md metadata.MD) *pb.Metadata {
	ret := &pb.Metadata{}
	for k, vv := range md {
		if metadataKeyOmit(k) {
			continue
		}
		for _, v := range vv {
			ret.Entry = append(ret.Entry,
				&pb.MetadataEntry{
					Key:   k,
					Value: []byte(v),
				},
			)
		}
	}
	return ret
}

func addrToProto(addr net.Addr) *pb.Address {
	ret := &pb.Address{}
	switch a := addr.(type) {
	case *net.TCPAddr:
		if a.IP.To4() != nil {
			ret.Type = pb.Address_TYPE_IPV4
		} else if a.IP.To16() != nil {
			ret.Type = pb.Address_TYPE_IPV6
		} else {
			ret.Type = pb.Address_TYPE_UNKNOWN
			// Do not set address and port fields.
			break
		}
		ret.Address = a.IP.String()
		ret.IpPort = uint32(a.Port)
	case *net.UnixAddr:
		ret.Type = pb.Address_TYPE_UNIX
		ret.Address = a.String()
	default:
		ret.Type = pb.Address_TYPE_UNKNOWN
	}
	return ret
}
