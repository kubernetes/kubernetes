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
	"context"
)

// THeaderProtocol is a thrift protocol that implements THeader:
// https://github.com/apache/thrift/blob/master/doc/specs/HeaderFormat.md
//
// It supports either binary or compact protocol as the wrapped protocol.
//
// Most of the THeader handlings are happening inside THeaderTransport.
type THeaderProtocol struct {
	transport *THeaderTransport

	// Will be initialized on first read/write.
	protocol TProtocol
}

// NewTHeaderProtocol creates a new THeaderProtocol from the underlying
// transport. The passed in transport will be wrapped with THeaderTransport.
//
// Note that THeaderTransport handles frame and zlib by itself,
// so the underlying transport should be a raw socket transports (TSocket or TSSLSocket),
// instead of rich transports like TZlibTransport or TFramedTransport.
func NewTHeaderProtocol(trans TTransport) *THeaderProtocol {
	t := NewTHeaderTransport(trans)
	p, _ := THeaderProtocolDefault.GetProtocol(t)
	return &THeaderProtocol{
		transport: t,
		protocol:  p,
	}
}

type tHeaderProtocolFactory struct{}

func (tHeaderProtocolFactory) GetProtocol(trans TTransport) TProtocol {
	return NewTHeaderProtocol(trans)
}

// NewTHeaderProtocolFactory creates a factory for THeader.
//
// It's a wrapper for NewTHeaderProtocol
func NewTHeaderProtocolFactory() TProtocolFactory {
	return tHeaderProtocolFactory{}
}

// Transport returns the underlying transport.
//
// It's guaranteed to be of type *THeaderTransport.
func (p *THeaderProtocol) Transport() TTransport {
	return p.transport
}

// GetReadHeaders returns the THeaderMap read from transport.
func (p *THeaderProtocol) GetReadHeaders() THeaderMap {
	return p.transport.GetReadHeaders()
}

// SetWriteHeader sets a header for write.
func (p *THeaderProtocol) SetWriteHeader(key, value string) {
	p.transport.SetWriteHeader(key, value)
}

// ClearWriteHeaders clears all write headers previously set.
func (p *THeaderProtocol) ClearWriteHeaders() {
	p.transport.ClearWriteHeaders()
}

// AddTransform add a transform for writing.
func (p *THeaderProtocol) AddTransform(transform THeaderTransformID) error {
	return p.transport.AddTransform(transform)
}

func (p *THeaderProtocol) Flush(ctx context.Context) error {
	return p.transport.Flush(ctx)
}

func (p *THeaderProtocol) WriteMessageBegin(name string, typeID TMessageType, seqID int32) error {
	newProto, err := p.transport.Protocol().GetProtocol(p.transport)
	if err != nil {
		return err
	}
	p.protocol = newProto
	p.transport.SequenceID = seqID
	return p.protocol.WriteMessageBegin(name, typeID, seqID)
}

func (p *THeaderProtocol) WriteMessageEnd() error {
	if err := p.protocol.WriteMessageEnd(); err != nil {
		return err
	}
	return p.transport.Flush(context.Background())
}

func (p *THeaderProtocol) WriteStructBegin(name string) error {
	return p.protocol.WriteStructBegin(name)
}

func (p *THeaderProtocol) WriteStructEnd() error {
	return p.protocol.WriteStructEnd()
}

func (p *THeaderProtocol) WriteFieldBegin(name string, typeID TType, id int16) error {
	return p.protocol.WriteFieldBegin(name, typeID, id)
}

func (p *THeaderProtocol) WriteFieldEnd() error {
	return p.protocol.WriteFieldEnd()
}

func (p *THeaderProtocol) WriteFieldStop() error {
	return p.protocol.WriteFieldStop()
}

func (p *THeaderProtocol) WriteMapBegin(keyType TType, valueType TType, size int) error {
	return p.protocol.WriteMapBegin(keyType, valueType, size)
}

func (p *THeaderProtocol) WriteMapEnd() error {
	return p.protocol.WriteMapEnd()
}

func (p *THeaderProtocol) WriteListBegin(elemType TType, size int) error {
	return p.protocol.WriteListBegin(elemType, size)
}

func (p *THeaderProtocol) WriteListEnd() error {
	return p.protocol.WriteListEnd()
}

func (p *THeaderProtocol) WriteSetBegin(elemType TType, size int) error {
	return p.protocol.WriteSetBegin(elemType, size)
}

func (p *THeaderProtocol) WriteSetEnd() error {
	return p.protocol.WriteSetEnd()
}

func (p *THeaderProtocol) WriteBool(value bool) error {
	return p.protocol.WriteBool(value)
}

func (p *THeaderProtocol) WriteByte(value int8) error {
	return p.protocol.WriteByte(value)
}

func (p *THeaderProtocol) WriteI16(value int16) error {
	return p.protocol.WriteI16(value)
}

func (p *THeaderProtocol) WriteI32(value int32) error {
	return p.protocol.WriteI32(value)
}

func (p *THeaderProtocol) WriteI64(value int64) error {
	return p.protocol.WriteI64(value)
}

func (p *THeaderProtocol) WriteDouble(value float64) error {
	return p.protocol.WriteDouble(value)
}

func (p *THeaderProtocol) WriteString(value string) error {
	return p.protocol.WriteString(value)
}

func (p *THeaderProtocol) WriteBinary(value []byte) error {
	return p.protocol.WriteBinary(value)
}

// ReadFrame calls underlying THeaderTransport's ReadFrame function.
func (p *THeaderProtocol) ReadFrame() error {
	return p.transport.ReadFrame()
}

func (p *THeaderProtocol) ReadMessageBegin() (name string, typeID TMessageType, seqID int32, err error) {
	if err = p.transport.ReadFrame(); err != nil {
		return
	}

	var newProto TProtocol
	newProto, err = p.transport.Protocol().GetProtocol(p.transport)
	if err != nil {
		tAppExc, ok := err.(TApplicationException)
		if !ok {
			return
		}
		if e := p.protocol.WriteMessageBegin("", EXCEPTION, seqID); e != nil {
			return
		}
		if e := tAppExc.Write(p.protocol); e != nil {
			return
		}
		if e := p.protocol.WriteMessageEnd(); e != nil {
			return
		}
		if e := p.transport.Flush(context.Background()); e != nil {
			return
		}
		return
	}
	p.protocol = newProto

	return p.protocol.ReadMessageBegin()
}

func (p *THeaderProtocol) ReadMessageEnd() error {
	return p.protocol.ReadMessageEnd()
}

func (p *THeaderProtocol) ReadStructBegin() (name string, err error) {
	return p.protocol.ReadStructBegin()
}

func (p *THeaderProtocol) ReadStructEnd() error {
	return p.protocol.ReadStructEnd()
}

func (p *THeaderProtocol) ReadFieldBegin() (name string, typeID TType, id int16, err error) {
	return p.protocol.ReadFieldBegin()
}

func (p *THeaderProtocol) ReadFieldEnd() error {
	return p.protocol.ReadFieldEnd()
}

func (p *THeaderProtocol) ReadMapBegin() (keyType TType, valueType TType, size int, err error) {
	return p.protocol.ReadMapBegin()
}

func (p *THeaderProtocol) ReadMapEnd() error {
	return p.protocol.ReadMapEnd()
}

func (p *THeaderProtocol) ReadListBegin() (elemType TType, size int, err error) {
	return p.protocol.ReadListBegin()
}

func (p *THeaderProtocol) ReadListEnd() error {
	return p.protocol.ReadListEnd()
}

func (p *THeaderProtocol) ReadSetBegin() (elemType TType, size int, err error) {
	return p.protocol.ReadSetBegin()
}

func (p *THeaderProtocol) ReadSetEnd() error {
	return p.protocol.ReadSetEnd()
}

func (p *THeaderProtocol) ReadBool() (value bool, err error) {
	return p.protocol.ReadBool()
}

func (p *THeaderProtocol) ReadByte() (value int8, err error) {
	return p.protocol.ReadByte()
}

func (p *THeaderProtocol) ReadI16() (value int16, err error) {
	return p.protocol.ReadI16()
}

func (p *THeaderProtocol) ReadI32() (value int32, err error) {
	return p.protocol.ReadI32()
}

func (p *THeaderProtocol) ReadI64() (value int64, err error) {
	return p.protocol.ReadI64()
}

func (p *THeaderProtocol) ReadDouble() (value float64, err error) {
	return p.protocol.ReadDouble()
}

func (p *THeaderProtocol) ReadString() (value string, err error) {
	return p.protocol.ReadString()
}

func (p *THeaderProtocol) ReadBinary() (value []byte, err error) {
	return p.protocol.ReadBinary()
}

func (p *THeaderProtocol) Skip(fieldType TType) error {
	return p.protocol.Skip(fieldType)
}
