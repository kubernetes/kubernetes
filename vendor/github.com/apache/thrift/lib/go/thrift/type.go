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

// Type constants in the Thrift protocol
type TType byte

const (
	STOP   = 0
	VOID   = 1
	BOOL   = 2
	BYTE   = 3
	I08    = 3
	DOUBLE = 4
	I16    = 6
	I32    = 8
	I64    = 10
	STRING = 11
	UTF7   = 11
	STRUCT = 12
	MAP    = 13
	SET    = 14
	LIST   = 15
	UTF8   = 16
	UTF16  = 17
	//BINARY = 18   wrong and unusued
)

var typeNames = map[int]string{
	STOP:   "STOP",
	VOID:   "VOID",
	BOOL:   "BOOL",
	BYTE:   "BYTE",
	DOUBLE: "DOUBLE",
	I16:    "I16",
	I32:    "I32",
	I64:    "I64",
	STRING: "STRING",
	STRUCT: "STRUCT",
	MAP:    "MAP",
	SET:    "SET",
	LIST:   "LIST",
	UTF8:   "UTF8",
	UTF16:  "UTF16",
}

func (p TType) String() string {
	if s, ok := typeNames[int(p)]; ok {
		return s
	}
	return "Unknown"
}
