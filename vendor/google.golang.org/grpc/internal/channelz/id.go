/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package channelz

import "fmt"

// Identifier is an opaque identifier which uniquely identifies an entity in the
// channelz database.
type Identifier struct {
	typ RefChannelType
	id  int64
	str string
	pid *Identifier
}

// Type returns the entity type corresponding to id.
func (id *Identifier) Type() RefChannelType {
	return id.typ
}

// Int returns the integer identifier corresponding to id.
func (id *Identifier) Int() int64 {
	return id.id
}

// String returns a string representation of the entity corresponding to id.
//
// This includes some information about the parent as well. Examples:
// Top-level channel: [Channel #channel-number]
// Nested channel:    [Channel #parent-channel-number Channel #channel-number]
// Sub channel:       [Channel #parent-channel SubChannel #subchannel-number]
func (id *Identifier) String() string {
	return id.str
}

// Equal returns true if other is the same as id.
func (id *Identifier) Equal(other *Identifier) bool {
	if (id != nil) != (other != nil) {
		return false
	}
	if id == nil && other == nil {
		return true
	}
	return id.typ == other.typ && id.id == other.id && id.pid == other.pid
}

// NewIdentifierForTesting returns a new opaque identifier to be used only for
// testing purposes.
func NewIdentifierForTesting(typ RefChannelType, id int64, pid *Identifier) *Identifier {
	return newIdentifer(typ, id, pid)
}

func newIdentifer(typ RefChannelType, id int64, pid *Identifier) *Identifier {
	str := fmt.Sprintf("%s #%d", typ, id)
	if pid != nil {
		str = fmt.Sprintf("%s %s", pid, str)
	}
	return &Identifier{typ: typ, id: id, str: str, pid: pid}
}
