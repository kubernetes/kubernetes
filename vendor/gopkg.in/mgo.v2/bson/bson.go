// BSON library for Go
//
// Copyright (c) 2010-2012 - Gustavo Niemeyer <gustavo@niemeyer.net>
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Package bson is an implementation of the BSON specification for Go:
//
//     http://bsonspec.org
//
// It was created as part of the mgo MongoDB driver for Go, but is standalone
// and may be used on its own without the driver.
package bson

import (
	"bytes"
	"crypto/md5"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// --------------------------------------------------------------------------
// The public API.

// A value implementing the bson.Getter interface will have its GetBSON
// method called when the given value has to be marshalled, and the result
// of this method will be marshaled in place of the actual object.
//
// If GetBSON returns return a non-nil error, the marshalling procedure
// will stop and error out with the provided value.
type Getter interface {
	GetBSON() (interface{}, error)
}

// A value implementing the bson.Setter interface will receive the BSON
// value via the SetBSON method during unmarshaling, and the object
// itself will not be changed as usual.
//
// If setting the value works, the method should return nil or alternatively
// bson.SetZero to set the respective field to its zero value (nil for
// pointer types). If SetBSON returns a value of type bson.TypeError, the
// BSON value will be omitted from a map or slice being decoded and the
// unmarshalling will continue. If it returns any other non-nil error, the
// unmarshalling procedure will stop and error out with the provided value.
//
// This interface is generally useful in pointer receivers, since the method
// will want to change the receiver. A type field that implements the Setter
// interface doesn't have to be a pointer, though.
//
// Unlike the usual behavior, unmarshalling onto a value that implements a
// Setter interface will NOT reset the value to its zero state. This allows
// the value to decide by itself how to be unmarshalled.
//
// For example:
//
//     type MyString string
//
//     func (s *MyString) SetBSON(raw bson.Raw) error {
//         return raw.Unmarshal(s)
//     }
//
type Setter interface {
	SetBSON(raw Raw) error
}

// SetZero may be returned from a SetBSON method to have the value set to
// its respective zero value. When used in pointer values, this will set the
// field to nil rather than to the pre-allocated value.
var SetZero = errors.New("set to zero")

// M is a convenient alias for a map[string]interface{} map, useful for
// dealing with BSON in a native way.  For instance:
//
//     bson.M{"a": 1, "b": true}
//
// There's no special handling for this type in addition to what's done anyway
// for an equivalent map type.  Elements in the map will be dumped in an
// undefined ordered. See also the bson.D type for an ordered alternative.
type M map[string]interface{}

// D represents a BSON document containing ordered elements. For example:
//
//     bson.D{{"a", 1}, {"b", true}}
//
// In some situations, such as when creating indexes for MongoDB, the order in
// which the elements are defined is important.  If the order is not important,
// using a map is generally more comfortable. See bson.M and bson.RawD.
type D []DocElem

// DocElem is an element of the bson.D document representation.
type DocElem struct {
	Name  string
	Value interface{}
}

// Map returns a map out of the ordered element name/value pairs in d.
func (d D) Map() (m M) {
	m = make(M, len(d))
	for _, item := range d {
		m[item.Name] = item.Value
	}
	return m
}

// The Raw type represents raw unprocessed BSON documents and elements.
// Kind is the kind of element as defined per the BSON specification, and
// Data is the raw unprocessed data for the respective element.
// Using this type it is possible to unmarshal or marshal values partially.
//
// Relevant documentation:
//
//     http://bsonspec.org/#/specification
//
type Raw struct {
	Kind byte
	Data []byte
}

// RawD represents a BSON document containing raw unprocessed elements.
// This low-level representation may be useful when lazily processing
// documents of uncertain content, or when manipulating the raw content
// documents in general.
type RawD []RawDocElem

// See the RawD type.
type RawDocElem struct {
	Name  string
	Value Raw
}

// ObjectId is a unique ID identifying a BSON value. It must be exactly 12 bytes
// long. MongoDB objects by default have such a property set in their "_id"
// property.
//
// http://www.mongodb.org/display/DOCS/Object+IDs
type ObjectId string

// ObjectIdHex returns an ObjectId from the provided hex representation.
// Calling this function with an invalid hex representation will
// cause a runtime panic. See the IsObjectIdHex function.
func ObjectIdHex(s string) ObjectId {
	d, err := hex.DecodeString(s)
	if err != nil || len(d) != 12 {
		panic(fmt.Sprintf("invalid input to ObjectIdHex: %q", s))
	}
	return ObjectId(d)
}

// IsObjectIdHex returns whether s is a valid hex representation of
// an ObjectId. See the ObjectIdHex function.
func IsObjectIdHex(s string) bool {
	if len(s) != 24 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// objectIdCounter is atomically incremented when generating a new ObjectId
// using NewObjectId() function. It's used as a counter part of an id.
var objectIdCounter uint32 = readRandomUint32()

// readRandomUint32 returns a random objectIdCounter.
func readRandomUint32() uint32 {
	var b [4]byte
	_, err := io.ReadFull(rand.Reader, b[:])
	if err != nil {
		panic(fmt.Errorf("cannot read random object id: %v", err))
	}
	return uint32((uint32(b[0]) << 0) | (uint32(b[1]) << 8) | (uint32(b[2]) << 16) | (uint32(b[3]) << 24))
}

// machineId stores machine id generated once and used in subsequent calls
// to NewObjectId function.
var machineId = readMachineId()
var processId = os.Getpid()

// readMachineId generates and returns a machine id.
// If this function fails to get the hostname it will cause a runtime error.
func readMachineId() []byte {
	var sum [3]byte
	id := sum[:]
	hostname, err1 := os.Hostname()
	if err1 != nil {
		_, err2 := io.ReadFull(rand.Reader, id)
		if err2 != nil {
			panic(fmt.Errorf("cannot get hostname: %v; %v", err1, err2))
		}
		return id
	}
	hw := md5.New()
	hw.Write([]byte(hostname))
	copy(id, hw.Sum(nil))
	return id
}

// NewObjectId returns a new unique ObjectId.
func NewObjectId() ObjectId {
	var b [12]byte
	// Timestamp, 4 bytes, big endian
	binary.BigEndian.PutUint32(b[:], uint32(time.Now().Unix()))
	// Machine, first 3 bytes of md5(hostname)
	b[4] = machineId[0]
	b[5] = machineId[1]
	b[6] = machineId[2]
	// Pid, 2 bytes, specs don't specify endianness, but we use big endian.
	b[7] = byte(processId >> 8)
	b[8] = byte(processId)
	// Increment, 3 bytes, big endian
	i := atomic.AddUint32(&objectIdCounter, 1)
	b[9] = byte(i >> 16)
	b[10] = byte(i >> 8)
	b[11] = byte(i)
	return ObjectId(b[:])
}

// NewObjectIdWithTime returns a dummy ObjectId with the timestamp part filled
// with the provided number of seconds from epoch UTC, and all other parts
// filled with zeroes. It's not safe to insert a document with an id generated
// by this method, it is useful only for queries to find documents with ids
// generated before or after the specified timestamp.
func NewObjectIdWithTime(t time.Time) ObjectId {
	var b [12]byte
	binary.BigEndian.PutUint32(b[:4], uint32(t.Unix()))
	return ObjectId(string(b[:]))
}

// String returns a hex string representation of the id.
// Example: ObjectIdHex("4d88e15b60f486e428412dc9").
func (id ObjectId) String() string {
	return fmt.Sprintf(`ObjectIdHex("%x")`, string(id))
}

// Hex returns a hex representation of the ObjectId.
func (id ObjectId) Hex() string {
	return hex.EncodeToString([]byte(id))
}

// MarshalJSON turns a bson.ObjectId into a json.Marshaller.
func (id ObjectId) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%x"`, string(id))), nil
}

var nullBytes = []byte("null")

// UnmarshalJSON turns *bson.ObjectId into a json.Unmarshaller.
func (id *ObjectId) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && (data[0] == '{' || data[0] == 'O') {
		var v struct {
			Id json.RawMessage `json:"$oid"`
			Func struct {
				Id json.RawMessage
			} `json:"$oidFunc"`
		}
		err := jdec(data, &v)
		if err == nil {
			if len(v.Id) > 0 {
				data = []byte(v.Id)
			} else {
				data = []byte(v.Func.Id)
			}
		}
	}
	if len(data) == 2 && data[0] == '"' && data[1] == '"' || bytes.Equal(data, nullBytes) {
		*id = ""
		return nil
	}
	if len(data) != 26 || data[0] != '"' || data[25] != '"' {
		return errors.New(fmt.Sprintf("invalid ObjectId in JSON: %s", string(data)))
	}
	var buf [12]byte
	_, err := hex.Decode(buf[:], data[1:25])
	if err != nil {
		return errors.New(fmt.Sprintf("invalid ObjectId in JSON: %s (%s)", string(data), err))
	}
	*id = ObjectId(string(buf[:]))
	return nil
}

// MarshalText turns bson.ObjectId into an encoding.TextMarshaler.
func (id ObjectId) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf("%x", string(id))), nil
}

// UnmarshalText turns *bson.ObjectId into an encoding.TextUnmarshaler.
func (id *ObjectId) UnmarshalText(data []byte) error {
	if len(data) == 1 && data[0] == ' ' || len(data) == 0 {
		*id = ""
		return nil
	}
	if len(data) != 24 {
		return fmt.Errorf("invalid ObjectId: %s", data)
	}
	var buf [12]byte
	_, err := hex.Decode(buf[:], data[:])
	if err != nil {
		return fmt.Errorf("invalid ObjectId: %s (%s)", data, err)
	}
	*id = ObjectId(string(buf[:]))
	return nil
}

// Valid returns true if id is valid. A valid id must contain exactly 12 bytes.
func (id ObjectId) Valid() bool {
	return len(id) == 12
}

// byteSlice returns byte slice of id from start to end.
// Calling this function with an invalid id will cause a runtime panic.
func (id ObjectId) byteSlice(start, end int) []byte {
	if len(id) != 12 {
		panic(fmt.Sprintf("invalid ObjectId: %q", string(id)))
	}
	return []byte(string(id)[start:end])
}

// Time returns the timestamp part of the id.
// It's a runtime error to call this method with an invalid id.
func (id ObjectId) Time() time.Time {
	// First 4 bytes of ObjectId is 32-bit big-endian seconds from epoch.
	secs := int64(binary.BigEndian.Uint32(id.byteSlice(0, 4)))
	return time.Unix(secs, 0)
}

// Machine returns the 3-byte machine id part of the id.
// It's a runtime error to call this method with an invalid id.
func (id ObjectId) Machine() []byte {
	return id.byteSlice(4, 7)
}

// Pid returns the process id part of the id.
// It's a runtime error to call this method with an invalid id.
func (id ObjectId) Pid() uint16 {
	return binary.BigEndian.Uint16(id.byteSlice(7, 9))
}

// Counter returns the incrementing value part of the id.
// It's a runtime error to call this method with an invalid id.
func (id ObjectId) Counter() int32 {
	b := id.byteSlice(9, 12)
	// Counter is stored as big-endian 3-byte value
	return int32(uint32(b[0])<<16 | uint32(b[1])<<8 | uint32(b[2]))
}

// The Symbol type is similar to a string and is used in languages with a
// distinct symbol type.
type Symbol string

// Now returns the current time with millisecond precision. MongoDB stores
// timestamps with the same precision, so a Time returned from this method
// will not change after a roundtrip to the database. That's the only reason
// why this function exists. Using the time.Now function also works fine
// otherwise.
func Now() time.Time {
	return time.Unix(0, time.Now().UnixNano()/1e6*1e6)
}

// MongoTimestamp is a special internal type used by MongoDB that for some
// strange reason has its own datatype defined in BSON.
type MongoTimestamp int64

type orderKey int64

// MaxKey is a special value that compares higher than all other possible BSON
// values in a MongoDB database.
var MaxKey = orderKey(1<<63 - 1)

// MinKey is a special value that compares lower than all other possible BSON
// values in a MongoDB database.
var MinKey = orderKey(-1 << 63)

type undefined struct{}

// Undefined represents the undefined BSON value.
var Undefined undefined

// Binary is a representation for non-standard binary values.  Any kind should
// work, but the following are known as of this writing:
//
//   0x00 - Generic. This is decoded as []byte(data), not Binary{0x00, data}.
//   0x01 - Function (!?)
//   0x02 - Obsolete generic.
//   0x03 - UUID
//   0x05 - MD5
//   0x80 - User defined.
//
type Binary struct {
	Kind byte
	Data []byte
}

// RegEx represents a regular expression.  The Options field may contain
// individual characters defining the way in which the pattern should be
// applied, and must be sorted. Valid options as of this writing are 'i' for
// case insensitive matching, 'm' for multi-line matching, 'x' for verbose
// mode, 'l' to make \w, \W, and similar be locale-dependent, 's' for dot-all
// mode (a '.' matches everything), and 'u' to make \w, \W, and similar match
// unicode. The value of the Options parameter is not verified before being
// marshaled into the BSON format.
type RegEx struct {
	Pattern string
	Options string
}

// JavaScript is a type that holds JavaScript code. If Scope is non-nil, it
// will be marshaled as a mapping from identifiers to values that may be
// used when evaluating the provided Code.
type JavaScript struct {
	Code  string
	Scope interface{}
}

// DBPointer refers to a document id in a namespace.
//
// This type is deprecated in the BSON specification and should not be used
// except for backwards compatibility with ancient applications.
type DBPointer struct {
	Namespace string
	Id        ObjectId
}

const initialBufferSize = 64

func handleErr(err *error) {
	if r := recover(); r != nil {
		if _, ok := r.(runtime.Error); ok {
			panic(r)
		} else if _, ok := r.(externalPanic); ok {
			panic(r)
		} else if s, ok := r.(string); ok {
			*err = errors.New(s)
		} else if e, ok := r.(error); ok {
			*err = e
		} else {
			panic(r)
		}
	}
}

// Marshal serializes the in value, which may be a map or a struct value.
// In the case of struct values, only exported fields will be serialized,
// and the order of serialized fields will match that of the struct itself.
// The lowercased field name is used as the key for each exported field,
// but this behavior may be changed using the respective field tag.
// The tag may also contain flags to tweak the marshalling behavior for
// the field. The tag formats accepted are:
//
//     "[<key>][,<flag1>[,<flag2>]]"
//
//     `(...) bson:"[<key>][,<flag1>[,<flag2>]]" (...)`
//
// The following flags are currently supported:
//
//     omitempty  Only include the field if it's not set to the zero
//                value for the type or to empty slices or maps.
//
//     minsize    Marshal an int64 value as an int32, if that's feasible
//                while preserving the numeric value.
//
//     inline     Inline the field, which must be a struct or a map,
//                causing all of its fields or keys to be processed as if
//                they were part of the outer struct. For maps, keys must
//                not conflict with the bson keys of other struct fields.
//
// Some examples:
//
//     type T struct {
//         A bool
//         B int    "myb"
//         C string "myc,omitempty"
//         D string `bson:",omitempty" json:"jsonkey"`
//         E int64  ",minsize"
//         F int64  "myf,omitempty,minsize"
//     }
//
func Marshal(in interface{}) (out []byte, err error) {
	defer handleErr(&err)
	e := &encoder{make([]byte, 0, initialBufferSize)}
	e.addDoc(reflect.ValueOf(in))
	return e.out, nil
}

// Unmarshal deserializes data from in into the out value.  The out value
// must be a map, a pointer to a struct, or a pointer to a bson.D value.
// In the case of struct values, only exported fields will be deserialized.
// The lowercased field name is used as the key for each exported field,
// but this behavior may be changed using the respective field tag.
// The tag may also contain flags to tweak the marshalling behavior for
// the field. The tag formats accepted are:
//
//     "[<key>][,<flag1>[,<flag2>]]"
//
//     `(...) bson:"[<key>][,<flag1>[,<flag2>]]" (...)`
//
// The following flags are currently supported during unmarshal (see the
// Marshal method for other flags):
//
//     inline     Inline the field, which must be a struct or a map.
//                Inlined structs are handled as if its fields were part
//                of the outer struct. An inlined map causes keys that do
//                not match any other struct field to be inserted in the
//                map rather than being discarded as usual.
//
// The target field or element types of out may not necessarily match
// the BSON values of the provided data.  The following conversions are
// made automatically:
//
// - Numeric types are converted if at least the integer part of the
//   value would be preserved correctly
// - Bools are converted to numeric types as 1 or 0
// - Numeric types are converted to bools as true if not 0 or false otherwise
// - Binary and string BSON data is converted to a string, array or byte slice
//
// If the value would not fit the type and cannot be converted, it's
// silently skipped.
//
// Pointer values are initialized when necessary.
func Unmarshal(in []byte, out interface{}) (err error) {
	if raw, ok := out.(*Raw); ok {
		raw.Kind = 3
		raw.Data = in
		return nil
	}
	defer handleErr(&err)
	v := reflect.ValueOf(out)
	switch v.Kind() {
	case reflect.Ptr:
		fallthrough
	case reflect.Map:
		d := newDecoder(in)
		d.readDocTo(v)
	case reflect.Struct:
		return errors.New("Unmarshal can't deal with struct values. Use a pointer.")
	default:
		return errors.New("Unmarshal needs a map or a pointer to a struct.")
	}
	return nil
}

// Unmarshal deserializes raw into the out value.  If the out value type
// is not compatible with raw, a *bson.TypeError is returned.
//
// See the Unmarshal function documentation for more details on the
// unmarshalling process.
func (raw Raw) Unmarshal(out interface{}) (err error) {
	defer handleErr(&err)
	v := reflect.ValueOf(out)
	switch v.Kind() {
	case reflect.Ptr:
		v = v.Elem()
		fallthrough
	case reflect.Map:
		d := newDecoder(raw.Data)
		good := d.readElemTo(v, raw.Kind)
		if !good {
			return &TypeError{v.Type(), raw.Kind}
		}
	case reflect.Struct:
		return errors.New("Raw Unmarshal can't deal with struct values. Use a pointer.")
	default:
		return errors.New("Raw Unmarshal needs a map or a valid pointer.")
	}
	return nil
}

type TypeError struct {
	Type reflect.Type
	Kind byte
}

func (e *TypeError) Error() string {
	return fmt.Sprintf("BSON kind 0x%02x isn't compatible with type %s", e.Kind, e.Type.String())
}

// --------------------------------------------------------------------------
// Maintain a mapping of keys to structure field indexes

type structInfo struct {
	FieldsMap  map[string]fieldInfo
	FieldsList []fieldInfo
	InlineMap  int
	Zero       reflect.Value
}

type fieldInfo struct {
	Key       string
	Num       int
	OmitEmpty bool
	MinSize   bool
	Inline    []int
}

var structMap = make(map[reflect.Type]*structInfo)
var structMapMutex sync.RWMutex

type externalPanic string

func (e externalPanic) String() string {
	return string(e)
}

func getStructInfo(st reflect.Type) (*structInfo, error) {
	structMapMutex.RLock()
	sinfo, found := structMap[st]
	structMapMutex.RUnlock()
	if found {
		return sinfo, nil
	}
	n := st.NumField()
	fieldsMap := make(map[string]fieldInfo)
	fieldsList := make([]fieldInfo, 0, n)
	inlineMap := -1
	for i := 0; i != n; i++ {
		field := st.Field(i)
		if field.PkgPath != "" && !field.Anonymous {
			continue // Private field
		}

		info := fieldInfo{Num: i}

		tag := field.Tag.Get("bson")
		if tag == "" && strings.Index(string(field.Tag), ":") < 0 {
			tag = string(field.Tag)
		}
		if tag == "-" {
			continue
		}

		inline := false
		fields := strings.Split(tag, ",")
		if len(fields) > 1 {
			for _, flag := range fields[1:] {
				switch flag {
				case "omitempty":
					info.OmitEmpty = true
				case "minsize":
					info.MinSize = true
				case "inline":
					inline = true
				default:
					msg := fmt.Sprintf("Unsupported flag %q in tag %q of type %s", flag, tag, st)
					panic(externalPanic(msg))
				}
			}
			tag = fields[0]
		}

		if inline {
			switch field.Type.Kind() {
			case reflect.Map:
				if inlineMap >= 0 {
					return nil, errors.New("Multiple ,inline maps in struct " + st.String())
				}
				if field.Type.Key() != reflect.TypeOf("") {
					return nil, errors.New("Option ,inline needs a map with string keys in struct " + st.String())
				}
				inlineMap = info.Num
			case reflect.Struct:
				sinfo, err := getStructInfo(field.Type)
				if err != nil {
					return nil, err
				}
				for _, finfo := range sinfo.FieldsList {
					if _, found := fieldsMap[finfo.Key]; found {
						msg := "Duplicated key '" + finfo.Key + "' in struct " + st.String()
						return nil, errors.New(msg)
					}
					if finfo.Inline == nil {
						finfo.Inline = []int{i, finfo.Num}
					} else {
						finfo.Inline = append([]int{i}, finfo.Inline...)
					}
					fieldsMap[finfo.Key] = finfo
					fieldsList = append(fieldsList, finfo)
				}
			default:
				panic("Option ,inline needs a struct value or map field")
			}
			continue
		}

		if tag != "" {
			info.Key = tag
		} else {
			info.Key = strings.ToLower(field.Name)
		}

		if _, found = fieldsMap[info.Key]; found {
			msg := "Duplicated key '" + info.Key + "' in struct " + st.String()
			return nil, errors.New(msg)
		}

		fieldsList = append(fieldsList, info)
		fieldsMap[info.Key] = info
	}
	sinfo = &structInfo{
		fieldsMap,
		fieldsList,
		inlineMap,
		reflect.New(st).Elem(),
	}
	structMapMutex.Lock()
	structMap[st] = sinfo
	structMapMutex.Unlock()
	return sinfo, nil
}
