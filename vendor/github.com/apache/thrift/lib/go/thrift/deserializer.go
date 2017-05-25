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

type TDeserializer struct {
	Transport TTransport
	Protocol  TProtocol
}

func NewTDeserializer() *TDeserializer {
	var transport TTransport
	transport = NewTMemoryBufferLen(1024)

	protocol := NewTBinaryProtocolFactoryDefault().GetProtocol(transport)

	return &TDeserializer{
		transport,
		protocol}
}

func (t *TDeserializer) ReadString(msg TStruct, s string) (err error) {
	err = nil
	if _, err = t.Transport.Write([]byte(s)); err != nil {
		return
	}
	if err = msg.Read(t.Protocol); err != nil {
		return
	}
	return
}

func (t *TDeserializer) Read(msg TStruct, b []byte) (err error) {
	err = nil
	if _, err = t.Transport.Write(b); err != nil {
		return
	}
	if err = msg.Read(t.Protocol); err != nil {
		return
	}
	return
}
