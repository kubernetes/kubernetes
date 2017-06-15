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

type TSerializer struct {
	Transport *TMemoryBuffer
	Protocol  TProtocol
}

type TStruct interface {
	Write(p TProtocol) error
	Read(p TProtocol) error
}

func NewTSerializer() *TSerializer {
	transport := NewTMemoryBufferLen(1024)
	protocol := NewTBinaryProtocolFactoryDefault().GetProtocol(transport)

	return &TSerializer{
		transport,
		protocol}
}

func (t *TSerializer) WriteString(msg TStruct) (s string, err error) {
	t.Transport.Reset()

	if err = msg.Write(t.Protocol); err != nil {
		return
	}

	if err = t.Protocol.Flush(); err != nil {
		return
	}
	if err = t.Transport.Flush(); err != nil {
		return
	}

	return t.Transport.String(), nil
}

func (t *TSerializer) Write(msg TStruct) (b []byte, err error) {
	t.Transport.Reset()

	if err = msg.Write(t.Protocol); err != nil {
		return
	}

	if err = t.Protocol.Flush(); err != nil {
		return
	}

	if err = t.Transport.Flush(); err != nil {
		return
	}

	b = append(b, t.Transport.Bytes()...)
	return
}
