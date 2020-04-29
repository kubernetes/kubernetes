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

// Helper class that encapsulates field metadata.
type field struct {
	name   string
	typeId TType
	id     int
}

func newField(n string, t TType, i int) *field {
	return &field{name: n, typeId: t, id: i}
}

func (p *field) Name() string {
	if p == nil {
		return ""
	}
	return p.name
}

func (p *field) TypeId() TType {
	if p == nil {
		return TType(VOID)
	}
	return p.typeId
}

func (p *field) Id() int {
	if p == nil {
		return -1
	}
	return p.id
}

func (p *field) String() string {
	if p == nil {
		return "<nil>"
	}
	return "<TField name:'" + p.name + "' type:" + string(p.typeId) + " field-id:" + string(p.id) + ">"
}

var ANONYMOUS_FIELD *field

type fieldSlice []field

func (p fieldSlice) Len() int {
	return len(p)
}

func (p fieldSlice) Less(i, j int) bool {
	return p[i].Id() < p[j].Id()
}

func (p fieldSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func init() {
	ANONYMOUS_FIELD = newField("", STOP, 0)
}
