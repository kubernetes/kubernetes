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

import "io"

type RichTransport struct {
	TTransport
}

// Wraps Transport to provide TRichTransport interface
func NewTRichTransport(trans TTransport) *RichTransport {
	return &RichTransport{trans}
}

func (r *RichTransport) ReadByte() (c byte, err error) {
	return readByte(r.TTransport)
}

func (r *RichTransport) WriteByte(c byte) error {
	return writeByte(r.TTransport, c)
}

func (r *RichTransport) WriteString(s string) (n int, err error) {
	return r.Write([]byte(s))
}

func (r *RichTransport) RemainingBytes() (num_bytes uint64) {
	return r.TTransport.RemainingBytes()
}

func readByte(r io.Reader) (c byte, err error) {
	v := [1]byte{0}
	n, err := r.Read(v[0:1])
	if n > 0 && (err == nil || err == io.EOF) {
		return v[0], nil
	}
	if n > 0 && err != nil {
		return v[0], err
	}
	if err != nil {
		return 0, err
	}
	return v[0], nil
}

func writeByte(w io.Writer, c byte) error {
	v := [1]byte{c}
	_, err := w.Write(v[0:1])
	return err
}

