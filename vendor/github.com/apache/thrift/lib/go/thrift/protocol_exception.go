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
	"encoding/base64"
)

// Thrift Protocol exception
type TProtocolException interface {
	TException
	TypeId() int
}

const (
	UNKNOWN_PROTOCOL_EXCEPTION = 0
	INVALID_DATA               = 1
	NEGATIVE_SIZE              = 2
	SIZE_LIMIT                 = 3
	BAD_VERSION                = 4
	NOT_IMPLEMENTED            = 5
	DEPTH_LIMIT                = 6
)

type tProtocolException struct {
	typeId  int
	message string
}

func (p *tProtocolException) TypeId() int {
	return p.typeId
}

func (p *tProtocolException) String() string {
	return p.message
}

func (p *tProtocolException) Error() string {
	return p.message
}

func NewTProtocolException(err error) TProtocolException {
	if err == nil {
		return nil
	}
	if e, ok := err.(TProtocolException); ok {
		return e
	}
	if _, ok := err.(base64.CorruptInputError); ok {
		return &tProtocolException{INVALID_DATA, err.Error()}
	}
	return &tProtocolException{UNKNOWN_PROTOCOL_EXCEPTION, err.Error()}
}

func NewTProtocolExceptionWithType(errType int, err error) TProtocolException {
	if err == nil {
		return nil
	}
	return &tProtocolException{errType, err.Error()}
}
