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
	"log"
)

type TDebugProtocol struct {
	Delegate  TProtocol
	LogPrefix string
}

type TDebugProtocolFactory struct {
	Underlying TProtocolFactory
	LogPrefix  string
}

func NewTDebugProtocolFactory(underlying TProtocolFactory, logPrefix string) *TDebugProtocolFactory {
	return &TDebugProtocolFactory{
		Underlying: underlying,
		LogPrefix:  logPrefix,
	}
}

func (t *TDebugProtocolFactory) GetProtocol(trans TTransport) TProtocol {
	return &TDebugProtocol{
		Delegate:  t.Underlying.GetProtocol(trans),
		LogPrefix: t.LogPrefix,
	}
}

func (tdp *TDebugProtocol) WriteMessageBegin(name string, typeId TMessageType, seqid int32) error {
	err := tdp.Delegate.WriteMessageBegin(name, typeId, seqid)
	log.Printf("%sWriteMessageBegin(name=%#v, typeId=%#v, seqid=%#v) => %#v", tdp.LogPrefix, name, typeId, seqid, err)
	return err
}
func (tdp *TDebugProtocol) WriteMessageEnd() error {
	err := tdp.Delegate.WriteMessageEnd()
	log.Printf("%sWriteMessageEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteStructBegin(name string) error {
	err := tdp.Delegate.WriteStructBegin(name)
	log.Printf("%sWriteStructBegin(name=%#v) => %#v", tdp.LogPrefix, name, err)
	return err
}
func (tdp *TDebugProtocol) WriteStructEnd() error {
	err := tdp.Delegate.WriteStructEnd()
	log.Printf("%sWriteStructEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteFieldBegin(name string, typeId TType, id int16) error {
	err := tdp.Delegate.WriteFieldBegin(name, typeId, id)
	log.Printf("%sWriteFieldBegin(name=%#v, typeId=%#v, id%#v) => %#v", tdp.LogPrefix, name, typeId, id, err)
	return err
}
func (tdp *TDebugProtocol) WriteFieldEnd() error {
	err := tdp.Delegate.WriteFieldEnd()
	log.Printf("%sWriteFieldEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteFieldStop() error {
	err := tdp.Delegate.WriteFieldStop()
	log.Printf("%sWriteFieldStop() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteMapBegin(keyType TType, valueType TType, size int) error {
	err := tdp.Delegate.WriteMapBegin(keyType, valueType, size)
	log.Printf("%sWriteMapBegin(keyType=%#v, valueType=%#v, size=%#v) => %#v", tdp.LogPrefix, keyType, valueType, size, err)
	return err
}
func (tdp *TDebugProtocol) WriteMapEnd() error {
	err := tdp.Delegate.WriteMapEnd()
	log.Printf("%sWriteMapEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteListBegin(elemType TType, size int) error {
	err := tdp.Delegate.WriteListBegin(elemType, size)
	log.Printf("%sWriteListBegin(elemType=%#v, size=%#v) => %#v", tdp.LogPrefix, elemType, size, err)
	return err
}
func (tdp *TDebugProtocol) WriteListEnd() error {
	err := tdp.Delegate.WriteListEnd()
	log.Printf("%sWriteListEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteSetBegin(elemType TType, size int) error {
	err := tdp.Delegate.WriteSetBegin(elemType, size)
	log.Printf("%sWriteSetBegin(elemType=%#v, size=%#v) => %#v", tdp.LogPrefix, elemType, size, err)
	return err
}
func (tdp *TDebugProtocol) WriteSetEnd() error {
	err := tdp.Delegate.WriteSetEnd()
	log.Printf("%sWriteSetEnd() => %#v", tdp.LogPrefix, err)
	return err
}
func (tdp *TDebugProtocol) WriteBool(value bool) error {
	err := tdp.Delegate.WriteBool(value)
	log.Printf("%sWriteBool(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteByte(value int8) error {
	err := tdp.Delegate.WriteByte(value)
	log.Printf("%sWriteByte(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteI16(value int16) error {
	err := tdp.Delegate.WriteI16(value)
	log.Printf("%sWriteI16(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteI32(value int32) error {
	err := tdp.Delegate.WriteI32(value)
	log.Printf("%sWriteI32(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteI64(value int64) error {
	err := tdp.Delegate.WriteI64(value)
	log.Printf("%sWriteI64(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteDouble(value float64) error {
	err := tdp.Delegate.WriteDouble(value)
	log.Printf("%sWriteDouble(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteString(value string) error {
	err := tdp.Delegate.WriteString(value)
	log.Printf("%sWriteString(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}
func (tdp *TDebugProtocol) WriteBinary(value []byte) error {
	err := tdp.Delegate.WriteBinary(value)
	log.Printf("%sWriteBinary(value=%#v) => %#v", tdp.LogPrefix, value, err)
	return err
}

func (tdp *TDebugProtocol) ReadMessageBegin() (name string, typeId TMessageType, seqid int32, err error) {
	name, typeId, seqid, err = tdp.Delegate.ReadMessageBegin()
	log.Printf("%sReadMessageBegin() (name=%#v, typeId=%#v, seqid=%#v, err=%#v)", tdp.LogPrefix, name, typeId, seqid, err)
	return
}
func (tdp *TDebugProtocol) ReadMessageEnd() (err error) {
	err = tdp.Delegate.ReadMessageEnd()
	log.Printf("%sReadMessageEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadStructBegin() (name string, err error) {
	name, err = tdp.Delegate.ReadStructBegin()
	log.Printf("%sReadStructBegin() (name%#v, err=%#v)", tdp.LogPrefix, name, err)
	return
}
func (tdp *TDebugProtocol) ReadStructEnd() (err error) {
	err = tdp.Delegate.ReadStructEnd()
	log.Printf("%sReadStructEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadFieldBegin() (name string, typeId TType, id int16, err error) {
	name, typeId, id, err = tdp.Delegate.ReadFieldBegin()
	log.Printf("%sReadFieldBegin() (name=%#v, typeId=%#v, id=%#v, err=%#v)", tdp.LogPrefix, name, typeId, id, err)
	return
}
func (tdp *TDebugProtocol) ReadFieldEnd() (err error) {
	err = tdp.Delegate.ReadFieldEnd()
	log.Printf("%sReadFieldEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadMapBegin() (keyType TType, valueType TType, size int, err error) {
	keyType, valueType, size, err = tdp.Delegate.ReadMapBegin()
	log.Printf("%sReadMapBegin() (keyType=%#v, valueType=%#v, size=%#v, err=%#v)", tdp.LogPrefix, keyType, valueType, size, err)
	return
}
func (tdp *TDebugProtocol) ReadMapEnd() (err error) {
	err = tdp.Delegate.ReadMapEnd()
	log.Printf("%sReadMapEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadListBegin() (elemType TType, size int, err error) {
	elemType, size, err = tdp.Delegate.ReadListBegin()
	log.Printf("%sReadListBegin() (elemType=%#v, size=%#v, err=%#v)", tdp.LogPrefix, elemType, size, err)
	return
}
func (tdp *TDebugProtocol) ReadListEnd() (err error) {
	err = tdp.Delegate.ReadListEnd()
	log.Printf("%sReadListEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadSetBegin() (elemType TType, size int, err error) {
	elemType, size, err = tdp.Delegate.ReadSetBegin()
	log.Printf("%sReadSetBegin() (elemType=%#v, size=%#v, err=%#v)", tdp.LogPrefix, elemType, size, err)
	return
}
func (tdp *TDebugProtocol) ReadSetEnd() (err error) {
	err = tdp.Delegate.ReadSetEnd()
	log.Printf("%sReadSetEnd() err=%#v", tdp.LogPrefix, err)
	return
}
func (tdp *TDebugProtocol) ReadBool() (value bool, err error) {
	value, err = tdp.Delegate.ReadBool()
	log.Printf("%sReadBool() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadByte() (value int8, err error) {
	value, err = tdp.Delegate.ReadByte()
	log.Printf("%sReadByte() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadI16() (value int16, err error) {
	value, err = tdp.Delegate.ReadI16()
	log.Printf("%sReadI16() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadI32() (value int32, err error) {
	value, err = tdp.Delegate.ReadI32()
	log.Printf("%sReadI32() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadI64() (value int64, err error) {
	value, err = tdp.Delegate.ReadI64()
	log.Printf("%sReadI64() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadDouble() (value float64, err error) {
	value, err = tdp.Delegate.ReadDouble()
	log.Printf("%sReadDouble() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadString() (value string, err error) {
	value, err = tdp.Delegate.ReadString()
	log.Printf("%sReadString() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) ReadBinary() (value []byte, err error) {
	value, err = tdp.Delegate.ReadBinary()
	log.Printf("%sReadBinary() (value=%#v, err=%#v)", tdp.LogPrefix, value, err)
	return
}
func (tdp *TDebugProtocol) Skip(fieldType TType) (err error) {
	err = tdp.Delegate.Skip(fieldType)
	log.Printf("%sSkip(fieldType=%#v) (err=%#v)", tdp.LogPrefix, fieldType, err)
	return
}
func (tdp *TDebugProtocol) Flush() (err error) {
	err = tdp.Delegate.Flush()
	log.Printf("%sFlush() (err=%#v)", tdp.LogPrefix, err)
	return
}

func (tdp *TDebugProtocol) Transport() TTransport {
	return tdp.Delegate.Transport()
}
