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
	"errors"
	"fmt"
)

const (
	VERSION_MASK = 0xffff0000
	VERSION_1    = 0x80010000
)

type TProtocol interface {
	WriteMessageBegin(name string, typeId TMessageType, seqid int32) error
	WriteMessageEnd() error
	WriteStructBegin(name string) error
	WriteStructEnd() error
	WriteFieldBegin(name string, typeId TType, id int16) error
	WriteFieldEnd() error
	WriteFieldStop() error
	WriteMapBegin(keyType TType, valueType TType, size int) error
	WriteMapEnd() error
	WriteListBegin(elemType TType, size int) error
	WriteListEnd() error
	WriteSetBegin(elemType TType, size int) error
	WriteSetEnd() error
	WriteBool(value bool) error
	WriteByte(value int8) error
	WriteI16(value int16) error
	WriteI32(value int32) error
	WriteI64(value int64) error
	WriteDouble(value float64) error
	WriteString(value string) error
	WriteBinary(value []byte) error

	ReadMessageBegin() (name string, typeId TMessageType, seqid int32, err error)
	ReadMessageEnd() error
	ReadStructBegin() (name string, err error)
	ReadStructEnd() error
	ReadFieldBegin() (name string, typeId TType, id int16, err error)
	ReadFieldEnd() error
	ReadMapBegin() (keyType TType, valueType TType, size int, err error)
	ReadMapEnd() error
	ReadListBegin() (elemType TType, size int, err error)
	ReadListEnd() error
	ReadSetBegin() (elemType TType, size int, err error)
	ReadSetEnd() error
	ReadBool() (value bool, err error)
	ReadByte() (value int8, err error)
	ReadI16() (value int16, err error)
	ReadI32() (value int32, err error)
	ReadI64() (value int64, err error)
	ReadDouble() (value float64, err error)
	ReadString() (value string, err error)
	ReadBinary() (value []byte, err error)

	Skip(fieldType TType) (err error)
	Flush(ctx context.Context) (err error)

	Transport() TTransport
}

// The maximum recursive depth the skip() function will traverse
const DEFAULT_RECURSION_DEPTH = 64

// Skips over the next data element from the provided input TProtocol object.
func SkipDefaultDepth(prot TProtocol, typeId TType) (err error) {
	return Skip(prot, typeId, DEFAULT_RECURSION_DEPTH)
}

// Skips over the next data element from the provided input TProtocol object.
func Skip(self TProtocol, fieldType TType, maxDepth int) (err error) {

	if maxDepth <= 0 {
		return NewTProtocolExceptionWithType(DEPTH_LIMIT, errors.New("Depth limit exceeded"))
	}

	switch fieldType {
	case BOOL:
		_, err = self.ReadBool()
		return
	case BYTE:
		_, err = self.ReadByte()
		return
	case I16:
		_, err = self.ReadI16()
		return
	case I32:
		_, err = self.ReadI32()
		return
	case I64:
		_, err = self.ReadI64()
		return
	case DOUBLE:
		_, err = self.ReadDouble()
		return
	case STRING:
		_, err = self.ReadString()
		return
	case STRUCT:
		if _, err = self.ReadStructBegin(); err != nil {
			return err
		}
		for {
			_, typeId, _, _ := self.ReadFieldBegin()
			if typeId == STOP {
				break
			}
			err := Skip(self, typeId, maxDepth-1)
			if err != nil {
				return err
			}
			self.ReadFieldEnd()
		}
		return self.ReadStructEnd()
	case MAP:
		keyType, valueType, size, err := self.ReadMapBegin()
		if err != nil {
			return err
		}
		for i := 0; i < size; i++ {
			err := Skip(self, keyType, maxDepth-1)
			if err != nil {
				return err
			}
			self.Skip(valueType)
		}
		return self.ReadMapEnd()
	case SET:
		elemType, size, err := self.ReadSetBegin()
		if err != nil {
			return err
		}
		for i := 0; i < size; i++ {
			err := Skip(self, elemType, maxDepth-1)
			if err != nil {
				return err
			}
		}
		return self.ReadSetEnd()
	case LIST:
		elemType, size, err := self.ReadListBegin()
		if err != nil {
			return err
		}
		for i := 0; i < size; i++ {
			err := Skip(self, elemType, maxDepth-1)
			if err != nil {
				return err
			}
		}
		return self.ReadListEnd()
	default:
		return NewTProtocolExceptionWithType(INVALID_DATA, errors.New(fmt.Sprintf("Unknown data type %d", fieldType)))
	}
	return nil
}
