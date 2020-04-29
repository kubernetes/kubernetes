/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package thrift

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
)

// Size in bytes for 32-bit ints.
const size32 = 4

type headerMeta struct {
	MagicFlags   uint32
	SequenceID   int32
	HeaderLength uint16
}

const headerMetaSize = 10

type clientType int

const (
	clientUnknown clientType = iota
	clientHeaders
	clientFramedBinary
	clientUnframedBinary
	clientFramedCompact
	clientUnframedCompact
)

// Constants defined in THeader format:
// https://github.com/apache/thrift/blob/master/doc/specs/HeaderFormat.md
const (
	THeaderHeaderMagic  uint32 = 0x0fff0000
	THeaderHeaderMask   uint32 = 0xffff0000
	THeaderFlagsMask    uint32 = 0x0000ffff
	THeaderMaxFrameSize uint32 = 0x3fffffff
)

// THeaderMap is the type of the header map in THeader transport.
type THeaderMap map[string]string

// THeaderProtocolID is the wrapped protocol id used in THeader.
type THeaderProtocolID int32

// Supported THeaderProtocolID values.
const (
	THeaderProtocolBinary  THeaderProtocolID = 0x00
	THeaderProtocolCompact THeaderProtocolID = 0x02
	THeaderProtocolDefault                   = THeaderProtocolBinary
)

// GetProtocol gets the corresponding TProtocol from the wrapped protocol id.
func (id THeaderProtocolID) GetProtocol(trans TTransport) (TProtocol, error) {
	switch id {
	default:
		return nil, NewTApplicationException(
			INVALID_PROTOCOL,
			fmt.Sprintf("THeader protocol id %d not supported", id),
		)
	case THeaderProtocolBinary:
		return NewTBinaryProtocolFactoryDefault().GetProtocol(trans), nil
	case THeaderProtocolCompact:
		return NewTCompactProtocol(trans), nil
	}
}

// THeaderTransformID defines the numeric id of the transform used.
type THeaderTransformID int32

// THeaderTransformID values
const (
	TransformNone THeaderTransformID = iota // 0, no special handling
	TransformZlib                           // 1, zlib
	// Rest of the values are not currently supported, namely HMAC and Snappy.
)

var supportedTransformIDs = map[THeaderTransformID]bool{
	TransformNone: true,
	TransformZlib: true,
}

// TransformReader is an io.ReadCloser that handles transforms reading.
type TransformReader struct {
	io.Reader

	closers []io.Closer
}

var _ io.ReadCloser = (*TransformReader)(nil)

// NewTransformReaderWithCapacity initializes a TransformReader with expected
// closers capacity.
//
// If you don't know the closers capacity beforehand, just use
//
//     &TransformReader{Reader: baseReader}
//
// instead would be sufficient.
func NewTransformReaderWithCapacity(baseReader io.Reader, capacity int) *TransformReader {
	return &TransformReader{
		Reader:  baseReader,
		closers: make([]io.Closer, 0, capacity),
	}
}

// Close calls the underlying closers in appropriate order,
// stops at and returns the first error encountered.
func (tr *TransformReader) Close() error {
	// Call closers in reversed order
	for i := len(tr.closers) - 1; i >= 0; i-- {
		if err := tr.closers[i].Close(); err != nil {
			return err
		}
	}
	return nil
}

// AddTransform adds a transform.
func (tr *TransformReader) AddTransform(id THeaderTransformID) error {
	switch id {
	default:
		return NewTApplicationException(
			INVALID_TRANSFORM,
			fmt.Sprintf("THeaderTransformID %d not supported", id),
		)
	case TransformNone:
		// no-op
	case TransformZlib:
		readCloser, err := zlib.NewReader(tr.Reader)
		if err != nil {
			return err
		}
		tr.Reader = readCloser
		tr.closers = append(tr.closers, readCloser)
	}
	return nil
}

// TransformWriter is an io.WriteCloser that handles transforms writing.
type TransformWriter struct {
	io.Writer

	closers []io.Closer
}

var _ io.WriteCloser = (*TransformWriter)(nil)

// NewTransformWriter creates a new TransformWriter with base writer and transforms.
func NewTransformWriter(baseWriter io.Writer, transforms []THeaderTransformID) (io.WriteCloser, error) {
	writer := &TransformWriter{
		Writer:  baseWriter,
		closers: make([]io.Closer, 0, len(transforms)),
	}
	for _, id := range transforms {
		if err := writer.AddTransform(id); err != nil {
			return nil, err
		}
	}
	return writer, nil
}

// Close calls the underlying closers in appropriate order,
// stops at and returns the first error encountered.
func (tw *TransformWriter) Close() error {
	// Call closers in reversed order
	for i := len(tw.closers) - 1; i >= 0; i-- {
		if err := tw.closers[i].Close(); err != nil {
			return err
		}
	}
	return nil
}

// AddTransform adds a transform.
func (tw *TransformWriter) AddTransform(id THeaderTransformID) error {
	switch id {
	default:
		return NewTApplicationException(
			INVALID_TRANSFORM,
			fmt.Sprintf("THeaderTransformID %d not supported", id),
		)
	case TransformNone:
		// no-op
	case TransformZlib:
		writeCloser := zlib.NewWriter(tw.Writer)
		tw.Writer = writeCloser
		tw.closers = append(tw.closers, writeCloser)
	}
	return nil
}

// THeaderInfoType is the type id of the info headers.
type THeaderInfoType int32

// Supported THeaderInfoType values.
const (
	_            THeaderInfoType = iota // Skip 0
	InfoKeyValue                        // 1
	// Rest of the info types are not supported.
)

// THeaderTransport is a Transport mode that implements THeader.
//
// Note that THeaderTransport handles frame and zlib by itself,
// so the underlying transport should be a raw socket transports (TSocket or TSSLSocket),
// instead of rich transports like TZlibTransport or TFramedTransport.
type THeaderTransport struct {
	SequenceID int32
	Flags      uint32

	transport TTransport

	// THeaderMap for read and write
	readHeaders  THeaderMap
	writeHeaders THeaderMap

	// Reading related variables.
	reader *bufio.Reader
	// When frame is detected, we read the frame fully into frameBuffer.
	frameBuffer bytes.Buffer
	// When it's non-nil, Read should read from frameReader instead of
	// reader, and EOF error indicates end of frame instead of end of all
	// transport.
	frameReader io.ReadCloser

	// Writing related variables
	writeBuffer     bytes.Buffer
	writeTransforms []THeaderTransformID

	clientType clientType
	protocolID THeaderProtocolID

	// buffer is used in the following scenarios to avoid repetitive
	// allocations, while 4 is big enough for all those scenarios:
	//
	// * header padding (max size 4)
	// * write the frame size (size 4)
	buffer [4]byte
}

var _ TTransport = (*THeaderTransport)(nil)

// NewTHeaderTransport creates THeaderTransport from the underlying transport.
//
// Please note that THeaderTransport handles framing and zlib by itself,
// so the underlying transport should be the raw socket transports (TSocket or TSSLSocket),
// instead of rich transports like TZlibTransport or TFramedTransport.
//
// If trans is already a *THeaderTransport, it will be returned as is.
func NewTHeaderTransport(trans TTransport) *THeaderTransport {
	if ht, ok := trans.(*THeaderTransport); ok {
		return ht
	}
	return &THeaderTransport{
		transport:    trans,
		reader:       bufio.NewReader(trans),
		writeHeaders: make(THeaderMap),
		protocolID:   THeaderProtocolDefault,
	}
}

// Open calls the underlying transport's Open function.
func (t *THeaderTransport) Open() error {
	return t.transport.Open()
}

// IsOpen calls the underlying transport's IsOpen function.
func (t *THeaderTransport) IsOpen() bool {
	return t.transport.IsOpen()
}

// ReadFrame tries to read the frame header, guess the client type, and handle
// unframed clients.
func (t *THeaderTransport) ReadFrame() error {
	if !t.needReadFrame() {
		// No need to read frame, skipping.
		return nil
	}
	// Peek and handle the first 32 bits.
	// They could either be the length field of a framed message,
	// or the first bytes of an unframed message.
	buf, err := t.reader.Peek(size32)
	if err != nil {
		return err
	}
	frameSize := binary.BigEndian.Uint32(buf)
	if frameSize&VERSION_MASK == VERSION_1 {
		t.clientType = clientUnframedBinary
		return nil
	}
	if buf[0] == COMPACT_PROTOCOL_ID && buf[1]&COMPACT_VERSION_MASK == COMPACT_VERSION {
		t.clientType = clientUnframedCompact
		return nil
	}

	// At this point it should be a framed message,
	// sanity check on frameSize then discard the peeked part.
	if frameSize > THeaderMaxFrameSize {
		return NewTProtocolExceptionWithType(
			SIZE_LIMIT,
			errors.New("frame too large"),
		)
	}
	t.reader.Discard(size32)

	// Read the frame fully into frameBuffer.
	_, err = io.Copy(
		&t.frameBuffer,
		io.LimitReader(t.reader, int64(frameSize)),
	)
	if err != nil {
		return err
	}
	t.frameReader = ioutil.NopCloser(&t.frameBuffer)

	// Peek and handle the next 32 bits.
	buf = t.frameBuffer.Bytes()[:size32]
	version := binary.BigEndian.Uint32(buf)
	if version&THeaderHeaderMask == THeaderHeaderMagic {
		t.clientType = clientHeaders
		return t.parseHeaders(frameSize)
	}
	if version&VERSION_MASK == VERSION_1 {
		t.clientType = clientFramedBinary
		return nil
	}
	if buf[0] == COMPACT_PROTOCOL_ID && buf[1]&COMPACT_VERSION_MASK == COMPACT_VERSION {
		t.clientType = clientFramedCompact
		return nil
	}
	if err := t.endOfFrame(); err != nil {
		return err
	}
	return NewTProtocolExceptionWithType(
		NOT_IMPLEMENTED,
		errors.New("unsupported client transport type"),
	)
}

// endOfFrame does end of frame handling.
//
// It closes frameReader, and also resets frame related states.
func (t *THeaderTransport) endOfFrame() error {
	defer func() {
		t.frameBuffer.Reset()
		t.frameReader = nil
	}()
	return t.frameReader.Close()
}

func (t *THeaderTransport) parseHeaders(frameSize uint32) error {
	if t.clientType != clientHeaders {
		return nil
	}

	var err error
	var meta headerMeta
	if err = binary.Read(&t.frameBuffer, binary.BigEndian, &meta); err != nil {
		return err
	}
	frameSize -= headerMetaSize
	t.Flags = meta.MagicFlags & THeaderFlagsMask
	t.SequenceID = meta.SequenceID
	headerLength := int64(meta.HeaderLength) * 4
	if int64(frameSize) < headerLength {
		return NewTProtocolExceptionWithType(
			SIZE_LIMIT,
			errors.New("header size is larger than the whole frame"),
		)
	}
	headerBuf := NewTMemoryBuffer()
	_, err = io.Copy(headerBuf, io.LimitReader(&t.frameBuffer, headerLength))
	if err != nil {
		return err
	}
	hp := NewTCompactProtocol(headerBuf)

	// At this point the header is already read into headerBuf,
	// and t.frameBuffer starts from the actual payload.
	protoID, err := hp.readVarint32()
	if err != nil {
		return err
	}
	t.protocolID = THeaderProtocolID(protoID)
	var transformCount int32
	transformCount, err = hp.readVarint32()
	if err != nil {
		return err
	}
	if transformCount > 0 {
		reader := NewTransformReaderWithCapacity(
			&t.frameBuffer,
			int(transformCount),
		)
		t.frameReader = reader
		transformIDs := make([]THeaderTransformID, transformCount)
		for i := 0; i < int(transformCount); i++ {
			id, err := hp.readVarint32()
			if err != nil {
				return err
			}
			transformIDs[i] = THeaderTransformID(id)
		}
		// The transform IDs on the wire was added based on the order of
		// writing, so on the reading side we need to reverse the order.
		for i := transformCount - 1; i >= 0; i-- {
			id := transformIDs[i]
			if err := reader.AddTransform(id); err != nil {
				return err
			}
		}
	}

	// The info part does not use the transforms yet, so it's
	// important to continue using headerBuf.
	headers := make(THeaderMap)
	for {
		infoType, err := hp.readVarint32()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if THeaderInfoType(infoType) == InfoKeyValue {
			count, err := hp.readVarint32()
			if err != nil {
				return err
			}
			for i := 0; i < int(count); i++ {
				key, err := hp.ReadString()
				if err != nil {
					return err
				}
				value, err := hp.ReadString()
				if err != nil {
					return err
				}
				headers[key] = value
			}
		} else {
			// Skip reading info section on the first
			// unsupported info type.
			break
		}
	}
	t.readHeaders = headers

	return nil
}

func (t *THeaderTransport) needReadFrame() bool {
	if t.clientType == clientUnknown {
		// This is a new connection that's never read before.
		return true
	}
	if t.isFramed() && t.frameReader == nil {
		// We just finished the last frame.
		return true
	}
	return false
}

func (t *THeaderTransport) Read(p []byte) (read int, err error) {
	err = t.ReadFrame()
	if err != nil {
		return
	}
	if t.frameReader != nil {
		read, err = t.frameReader.Read(p)
		if err == io.EOF {
			err = t.endOfFrame()
			if err != nil {
				return
			}
			if read < len(p) {
				var nextRead int
				nextRead, err = t.Read(p[read:])
				read += nextRead
			}
		}
		return
	}
	return t.reader.Read(p)
}

// Write writes data to the write buffer.
//
// You need to call Flush to actually write them to the transport.
func (t *THeaderTransport) Write(p []byte) (int, error) {
	return t.writeBuffer.Write(p)
}

// Flush writes the appropriate header and the write buffer to the underlying transport.
func (t *THeaderTransport) Flush(ctx context.Context) error {
	if t.writeBuffer.Len() == 0 {
		return nil
	}

	defer t.writeBuffer.Reset()

	switch t.clientType {
	default:
		fallthrough
	case clientUnknown:
		t.clientType = clientHeaders
		fallthrough
	case clientHeaders:
		headers := NewTMemoryBuffer()
		hp := NewTCompactProtocol(headers)
		if _, err := hp.writeVarint32(int32(t.protocolID)); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		if _, err := hp.writeVarint32(int32(len(t.writeTransforms))); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		for _, transform := range t.writeTransforms {
			if _, err := hp.writeVarint32(int32(transform)); err != nil {
				return NewTTransportExceptionFromError(err)
			}
		}
		if len(t.writeHeaders) > 0 {
			if _, err := hp.writeVarint32(int32(InfoKeyValue)); err != nil {
				return NewTTransportExceptionFromError(err)
			}
			if _, err := hp.writeVarint32(int32(len(t.writeHeaders))); err != nil {
				return NewTTransportExceptionFromError(err)
			}
			for key, value := range t.writeHeaders {
				if err := hp.WriteString(key); err != nil {
					return NewTTransportExceptionFromError(err)
				}
				if err := hp.WriteString(value); err != nil {
					return NewTTransportExceptionFromError(err)
				}
			}
		}
		padding := 4 - headers.Len()%4
		if padding < 4 {
			buf := t.buffer[:padding]
			for i := range buf {
				buf[i] = 0
			}
			if _, err := headers.Write(buf); err != nil {
				return NewTTransportExceptionFromError(err)
			}
		}

		var payload bytes.Buffer
		meta := headerMeta{
			MagicFlags:   THeaderHeaderMagic + t.Flags&THeaderFlagsMask,
			SequenceID:   t.SequenceID,
			HeaderLength: uint16(headers.Len() / 4),
		}
		if err := binary.Write(&payload, binary.BigEndian, meta); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		if _, err := io.Copy(&payload, headers); err != nil {
			return NewTTransportExceptionFromError(err)
		}

		writer, err := NewTransformWriter(&payload, t.writeTransforms)
		if err != nil {
			return NewTTransportExceptionFromError(err)
		}
		if _, err := io.Copy(writer, &t.writeBuffer); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		if err := writer.Close(); err != nil {
			return NewTTransportExceptionFromError(err)
		}

		// First write frame length
		buf := t.buffer[:size32]
		binary.BigEndian.PutUint32(buf, uint32(payload.Len()))
		if _, err := t.transport.Write(buf); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		// Then write the payload
		if _, err := io.Copy(t.transport, &payload); err != nil {
			return NewTTransportExceptionFromError(err)
		}

	case clientFramedBinary, clientFramedCompact:
		buf := t.buffer[:size32]
		binary.BigEndian.PutUint32(buf, uint32(t.writeBuffer.Len()))
		if _, err := t.transport.Write(buf); err != nil {
			return NewTTransportExceptionFromError(err)
		}
		fallthrough
	case clientUnframedBinary, clientUnframedCompact:
		if _, err := io.Copy(t.transport, &t.writeBuffer); err != nil {
			return NewTTransportExceptionFromError(err)
		}
	}

	select {
	default:
	case <-ctx.Done():
		return NewTTransportExceptionFromError(ctx.Err())
	}

	return t.transport.Flush(ctx)
}

// Close closes the transport, along with its underlying transport.
func (t *THeaderTransport) Close() error {
	if err := t.Flush(context.Background()); err != nil {
		return err
	}
	return t.transport.Close()
}

// RemainingBytes calls underlying transport's RemainingBytes.
//
// Even in framed cases, because of all the possible compression transforms
// involved, the remaining frame size is likely to be different from the actual
// remaining readable bytes, so we don't bother to keep tracking the remaining
// frame size by ourselves and just use the underlying transport's
// RemainingBytes directly.
func (t *THeaderTransport) RemainingBytes() uint64 {
	return t.transport.RemainingBytes()
}

// GetReadHeaders returns the THeaderMap read from transport.
func (t *THeaderTransport) GetReadHeaders() THeaderMap {
	return t.readHeaders
}

// SetWriteHeader sets a header for write.
func (t *THeaderTransport) SetWriteHeader(key, value string) {
	t.writeHeaders[key] = value
}

// ClearWriteHeaders clears all write headers previously set.
func (t *THeaderTransport) ClearWriteHeaders() {
	t.writeHeaders = make(THeaderMap)
}

// AddTransform add a transform for writing.
func (t *THeaderTransport) AddTransform(transform THeaderTransformID) error {
	if !supportedTransformIDs[transform] {
		return NewTProtocolExceptionWithType(
			NOT_IMPLEMENTED,
			fmt.Errorf("THeaderTransformID %d not supported", transform),
		)
	}
	t.writeTransforms = append(t.writeTransforms, transform)
	return nil
}

// Protocol returns the wrapped protocol id used in this THeaderTransport.
func (t *THeaderTransport) Protocol() THeaderProtocolID {
	switch t.clientType {
	default:
		return t.protocolID
	case clientFramedBinary, clientUnframedBinary:
		return THeaderProtocolBinary
	case clientFramedCompact, clientUnframedCompact:
		return THeaderProtocolCompact
	}
}

func (t *THeaderTransport) isFramed() bool {
	switch t.clientType {
	default:
		return false
	case clientHeaders, clientFramedBinary, clientFramedCompact:
		return true
	}
}

// THeaderTransportFactory is a TTransportFactory implementation to create
// THeaderTransport.
type THeaderTransportFactory struct {
	// The underlying factory, could be nil.
	Factory TTransportFactory
}

// NewTHeaderTransportFactory creates a new *THeaderTransportFactory.
func NewTHeaderTransportFactory(factory TTransportFactory) TTransportFactory {
	return &THeaderTransportFactory{
		Factory: factory,
	}
}

// GetTransport implements TTransportFactory.
func (f *THeaderTransportFactory) GetTransport(trans TTransport) (TTransport, error) {
	if f.Factory != nil {
		t, err := f.Factory.GetTransport(trans)
		if err != nil {
			return nil, err
		}
		return NewTHeaderTransport(t), nil
	}
	return NewTHeaderTransport(trans), nil
}
