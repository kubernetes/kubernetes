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

const (
	UNKNOWN_APPLICATION_EXCEPTION  = 0
	UNKNOWN_METHOD                 = 1
	INVALID_MESSAGE_TYPE_EXCEPTION = 2
	WRONG_METHOD_NAME              = 3
	BAD_SEQUENCE_ID                = 4
	MISSING_RESULT                 = 5
	INTERNAL_ERROR                 = 6
	PROTOCOL_ERROR                 = 7
)

// Application level Thrift exception
type TApplicationException interface {
	TException
	TypeId() int32
	Read(iprot TProtocol) (TApplicationException, error)
	Write(oprot TProtocol) error
}

type tApplicationException struct {
	message string
	type_   int32
}

func (e tApplicationException) Error() string {
	return e.message
}

func NewTApplicationException(type_ int32, message string) TApplicationException {
	return &tApplicationException{message, type_}
}

func (p *tApplicationException) TypeId() int32 {
	return p.type_
}

func (p *tApplicationException) Read(iprot TProtocol) (TApplicationException, error) {
	_, err := iprot.ReadStructBegin()
	if err != nil {
		return nil, err
	}

	message := ""
	type_ := int32(UNKNOWN_APPLICATION_EXCEPTION)

	for {
		_, ttype, id, err := iprot.ReadFieldBegin()
		if err != nil {
			return nil, err
		}
		if ttype == STOP {
			break
		}
		switch id {
		case 1:
			if ttype == STRING {
				if message, err = iprot.ReadString(); err != nil {
					return nil, err
				}
			} else {
				if err = SkipDefaultDepth(iprot, ttype); err != nil {
					return nil, err
				}
			}
		case 2:
			if ttype == I32 {
				if type_, err = iprot.ReadI32(); err != nil {
					return nil, err
				}
			} else {
				if err = SkipDefaultDepth(iprot, ttype); err != nil {
					return nil, err
				}
			}
		default:
			if err = SkipDefaultDepth(iprot, ttype); err != nil {
				return nil, err
			}
		}
		if err = iprot.ReadFieldEnd(); err != nil {
			return nil, err
		}
	}
	return NewTApplicationException(type_, message), iprot.ReadStructEnd()
}

func (p *tApplicationException) Write(oprot TProtocol) (err error) {
	err = oprot.WriteStructBegin("TApplicationException")
	if len(p.Error()) > 0 {
		err = oprot.WriteFieldBegin("message", STRING, 1)
		if err != nil {
			return
		}
		err = oprot.WriteString(p.Error())
		if err != nil {
			return
		}
		err = oprot.WriteFieldEnd()
		if err != nil {
			return
		}
	}
	err = oprot.WriteFieldBegin("type", I32, 2)
	if err != nil {
		return
	}
	err = oprot.WriteI32(p.type_)
	if err != nil {
		return
	}
	err = oprot.WriteFieldEnd()
	if err != nil {
		return
	}
	err = oprot.WriteFieldStop()
	if err != nil {
		return
	}
	err = oprot.WriteStructEnd()
	return
}
