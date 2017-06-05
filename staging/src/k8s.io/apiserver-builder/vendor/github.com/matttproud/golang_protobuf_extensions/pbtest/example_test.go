// Copyright 2015 Matt T. Proud
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pbtest

import (
	"testing"
	"testing/quick"

	"github.com/golang/protobuf/proto"
)

func Example() {
	// You would place this in a top-level function, like TestDatastore(t *testing.T).
	var (
		datastore Datastore
		t         *testing.T
	)
	if err := quick.Check(func(rec *CustomerRecord) bool {
		// testing/quick generated rec using quick.Value.  We want to ensure that
		// semi-internal struct fields are recursively reset to a known value.
		SanitizeGenerated(rec)
		// Ensure that any record can be stored, no matter what!
		if err := datastore.Store(rec); err != nil {
			return false
		}
		return true
	}, nil); err != nil {
		t.Fatal(err)
	}
}

// Datastore models some system under test.
type Datastore struct{}

// Store stores a customer record.
func (Datastore) Store(*CustomerRecord) error { return nil }

// Types below are generated from protoc --go_out=. example.proto, where
// example.proto contains
// """
// syntax = "proto2";
// message CustomerRecord {
// }
// """

type CustomerRecord struct {
	XXX_unrecognized []byte `json:"-"`
}

func (m *CustomerRecord) Reset()         { *m = CustomerRecord{} }
func (m *CustomerRecord) String() string { return proto.CompactTextString(m) }
func (*CustomerRecord) ProtoMessage()    {}
