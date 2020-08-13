// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"database/sql/driver"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	bsonprim "go.mongodb.org/mongo-driver/bson/primitive"
)

func init() {
	var id ObjectId
	// register this format in the default registry
	Default.Add("bsonobjectid", &id, IsBSONObjectID)
}

// IsBSONObjectID returns true when the string is a valid BSON.ObjectId
func IsBSONObjectID(str string) bool {
	_, err := bsonprim.ObjectIDFromHex(str)
	return err == nil
}

// ObjectId represents a BSON object ID (alias to go.mongodb.org/mongo-driver/bson/primitive.ObjectID)
//
// swagger:strfmt bsonobjectid
type ObjectId bsonprim.ObjectID

// NewObjectId creates a ObjectId from a Hex String
func NewObjectId(hex string) ObjectId {
	oid, err := bsonprim.ObjectIDFromHex(hex)
	if err != nil {
		panic(err)
	}
	return ObjectId(oid)
}

// MarshalText turns this instance into text
func (id ObjectId) MarshalText() ([]byte, error) {
	oid := bsonprim.ObjectID(id)
	if oid == bsonprim.NilObjectID {
		return nil, nil
	}
	return []byte(oid.Hex()), nil
}

// UnmarshalText hydrates this instance from text
func (id *ObjectId) UnmarshalText(data []byte) error { // validation is performed later on
	if len(data) == 0 {
		*id = ObjectId(bsonprim.NilObjectID)
		return nil
	}
	oidstr := string(data)
	oid, err := bsonprim.ObjectIDFromHex(oidstr)
	if err != nil {
		return err
	}
	*id = ObjectId(oid)
	return nil
}

// Scan read a value from a database driver
func (id *ObjectId) Scan(raw interface{}) error {
	var data []byte
	switch v := raw.(type) {
	case []byte:
		data = v
	case string:
		data = []byte(v)
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.URI from: %#v", v)
	}

	return id.UnmarshalText(data)
}

// Value converts a value to a database driver value
func (id ObjectId) Value() (driver.Value, error) {
	return driver.Value(bsonprim.ObjectID(id).Hex()), nil
}

func (id ObjectId) String() string {
	return bsonprim.ObjectID(id).String()
}

// MarshalJSON returns the ObjectId as JSON
func (id ObjectId) MarshalJSON() ([]byte, error) {
	return bsonprim.ObjectID(id).MarshalJSON()
}

// UnmarshalJSON sets the ObjectId from JSON
func (id *ObjectId) UnmarshalJSON(data []byte) error {
	var obj bsonprim.ObjectID
	if err := obj.UnmarshalJSON(data); err != nil {
		return err
	}
	*id = ObjectId(obj)
	return nil
}

// MarshalBSON renders the object id as a BSON document
func (id ObjectId) MarshalBSON() ([]byte, error) {
	return bson.Marshal(bson.M{"data": bsonprim.ObjectID(id)})
}

// UnmarshalBSON reads the objectId from a BSON document
func (id *ObjectId) UnmarshalBSON(data []byte) error {
	var obj struct {
		Data bsonprim.ObjectID
	}
	if err := bson.Unmarshal(data, &obj); err != nil {
		return err
	}
	*id = ObjectId(obj.Data)
	return nil
}

// MarshalBSONValue is an interface implemented by types that can marshal themselves
// into a BSON document represented as bytes. The bytes returned must be a valid
// BSON document if the error is nil.
func (id ObjectId) MarshalBSONValue() (bsontype.Type, []byte, error) {
	oid := bsonprim.ObjectID(id)
	return bsontype.ObjectID, oid[:], nil
}

// UnmarshalBSONValue is an interface implemented by types that can unmarshal a
// BSON value representation of themselves. The BSON bytes and type can be
// assumed to be valid. UnmarshalBSONValue must copy the BSON value bytes if it
// wishes to retain the data after returning.
func (id *ObjectId) UnmarshalBSONValue(tpe bsontype.Type, data []byte) error {
	var oid bsonprim.ObjectID
	copy(oid[:], data)
	*id = ObjectId(oid)
	return nil
}

// DeepCopyInto copies the receiver and writes its value into out.
func (id *ObjectId) DeepCopyInto(out *ObjectId) {
	*out = *id
}

// DeepCopy copies the receiver into a new ObjectId.
func (id *ObjectId) DeepCopy() *ObjectId {
	if id == nil {
		return nil
	}
	out := new(ObjectId)
	id.DeepCopyInto(out)
	return out
}
