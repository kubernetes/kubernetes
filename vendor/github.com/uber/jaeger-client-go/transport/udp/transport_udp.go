// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package udp

import (
	"errors"

	"github.com/apache/thrift/lib/go/thrift"

	"github.com/uber/jaeger-client-go/thrift-gen/zipkincore"
	"github.com/uber/jaeger-client-go/transport"
	"github.com/uber/jaeger-client-go/utils"
)

// Empirically obtained constant for how many bytes in the message are used for envelope.
// The total datagram size is sizeof(Span) * numSpans + emitSpanBatchOverhead <= maxPacketSize
// There is a unit test `TestEmitSpanBatchOverhead` that validates this number.
// Note that due to the use of Compact Thrift protocol, overhead grows with the number of spans
// in the batch, because the length of the list is encoded as varint32, as well as SeqId.
const emitSpanBatchOverhead = 30

const defaultUDPSpanServerHostPort = "localhost:5775"

var errSpanTooLarge = errors.New("Span is too large")

type udpSender struct {
	client         *utils.AgentClientUDP
	maxPacketSize  int                   // max size of datagram in bytes
	maxSpanBytes   int                   // max number of bytes to record spans (excluding envelope) in the datagram
	byteBufferSize int                   // current number of span bytes accumulated in the buffer
	spanBuffer     []*zipkincore.Span    // spans buffered before a flush
	thriftBuffer   *thrift.TMemoryBuffer // buffer used to calculate byte size of a span
	thriftProtocol thrift.TProtocol
}

// NewUDPTransport creates a reporter that submits spans to jaeger-agent
func NewUDPTransport(hostPort string, maxPacketSize int) (transport.Transport, error) {
	if len(hostPort) == 0 {
		hostPort = defaultUDPSpanServerHostPort
	}
	if maxPacketSize == 0 {
		maxPacketSize = utils.UDPPacketMaxLength
	}

	protocolFactory := thrift.NewTCompactProtocolFactory()

	// Each span is first written to thriftBuffer to determine its size in bytes.
	thriftBuffer := thrift.NewTMemoryBufferLen(maxPacketSize)
	thriftProtocol := protocolFactory.GetProtocol(thriftBuffer)

	client, err := utils.NewAgentClientUDP(hostPort, maxPacketSize)
	if err != nil {
		return nil, err
	}

	sender := &udpSender{
		client:         client,
		maxSpanBytes:   maxPacketSize - emitSpanBatchOverhead,
		thriftBuffer:   thriftBuffer,
		thriftProtocol: thriftProtocol}
	return sender, nil
}

func (s *udpSender) calcSpanSize(span *zipkincore.Span) (int, error) {
	s.thriftBuffer.Reset()
	if err := span.Write(s.thriftProtocol); err != nil {
		return 0, err
	}
	return s.thriftBuffer.Len(), nil
}

func (s *udpSender) Append(span *zipkincore.Span) (int, error) {
	spanSize, err := s.calcSpanSize(span)
	if err != nil {
		// should not be getting this error from in-memory transport - ¯\_(ツ)_/¯
		return 1, err
	}
	if spanSize > s.maxSpanBytes {
		return 1, errSpanTooLarge
	}

	s.byteBufferSize += spanSize
	if s.byteBufferSize <= s.maxSpanBytes {
		s.spanBuffer = append(s.spanBuffer, span)
		if s.byteBufferSize < s.maxSpanBytes {
			return 0, nil
		}
		return s.Flush()
	}
	// the latest span did not fit in the buffer
	n, err := s.Flush()
	s.spanBuffer = append(s.spanBuffer, span)
	s.byteBufferSize = spanSize
	return n, err
}

func (s *udpSender) Flush() (int, error) {
	n := len(s.spanBuffer)
	if n == 0 {
		return 0, nil
	}
	err := s.client.EmitZipkinBatch(s.spanBuffer)
	s.resetBuffers()

	return n, err
}

func (s *udpSender) Close() error {
	return s.client.Close()
}

func (s *udpSender) resetBuffers() {
	for i := range s.spanBuffer {
		s.spanBuffer[i] = nil
	}
	s.spanBuffer = s.spanBuffer[:0]
	s.byteBufferSize = 0
}
