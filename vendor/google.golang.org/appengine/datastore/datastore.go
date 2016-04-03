// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/datastore"
)

var (
	// ErrInvalidEntityType is returned when functions like Get or Next are
	// passed a dst or src argument of invalid type.
	ErrInvalidEntityType = errors.New("datastore: invalid entity type")
	// ErrInvalidKey is returned when an invalid key is presented.
	ErrInvalidKey = errors.New("datastore: invalid key")
	// ErrNoSuchEntity is returned when no entity was found for a given key.
	ErrNoSuchEntity = errors.New("datastore: no such entity")
)

// ErrFieldMismatch is returned when a field is to be loaded into a different
// type than the one it was stored from, or when a field is missing or
// unexported in the destination struct.
// StructType is the type of the struct pointed to by the destination argument
// passed to Get or to Iterator.Next.
type ErrFieldMismatch struct {
	StructType reflect.Type
	FieldName  string
	Reason     string
}

func (e *ErrFieldMismatch) Error() string {
	return fmt.Sprintf("datastore: cannot load field %q into a %q: %s",
		e.FieldName, e.StructType, e.Reason)
}

// protoToKey converts a Reference proto to a *Key.
func protoToKey(r *pb.Reference) (k *Key, err error) {
	appID := r.GetApp()
	namespace := r.GetNameSpace()
	for _, e := range r.Path.Element {
		k = &Key{
			kind:      e.GetType(),
			stringID:  e.GetName(),
			intID:     e.GetId(),
			parent:    k,
			appID:     appID,
			namespace: namespace,
		}
		if !k.valid() {
			return nil, ErrInvalidKey
		}
	}
	return
}

// keyToProto converts a *Key to a Reference proto.
func keyToProto(defaultAppID string, k *Key) *pb.Reference {
	appID := k.appID
	if appID == "" {
		appID = defaultAppID
	}
	n := 0
	for i := k; i != nil; i = i.parent {
		n++
	}
	e := make([]*pb.Path_Element, n)
	for i := k; i != nil; i = i.parent {
		n--
		e[n] = &pb.Path_Element{
			Type: &i.kind,
		}
		// At most one of {Name,Id} should be set.
		// Neither will be set for incomplete keys.
		if i.stringID != "" {
			e[n].Name = &i.stringID
		} else if i.intID != 0 {
			e[n].Id = &i.intID
		}
	}
	var namespace *string
	if k.namespace != "" {
		namespace = proto.String(k.namespace)
	}
	return &pb.Reference{
		App:       proto.String(appID),
		NameSpace: namespace,
		Path: &pb.Path{
			Element: e,
		},
	}
}

// multiKeyToProto is a batch version of keyToProto.
func multiKeyToProto(appID string, key []*Key) []*pb.Reference {
	ret := make([]*pb.Reference, len(key))
	for i, k := range key {
		ret[i] = keyToProto(appID, k)
	}
	return ret
}

// multiValid is a batch version of Key.valid. It returns an error, not a
// []bool.
func multiValid(key []*Key) error {
	invalid := false
	for _, k := range key {
		if !k.valid() {
			invalid = true
			break
		}
	}
	if !invalid {
		return nil
	}
	err := make(appengine.MultiError, len(key))
	for i, k := range key {
		if !k.valid() {
			err[i] = ErrInvalidKey
		}
	}
	return err
}

// It's unfortunate that the two semantically equivalent concepts pb.Reference
// and pb.PropertyValue_ReferenceValue aren't the same type. For example, the
// two have different protobuf field numbers.

// referenceValueToKey is the same as protoToKey except the input is a
// PropertyValue_ReferenceValue instead of a Reference.
func referenceValueToKey(r *pb.PropertyValue_ReferenceValue) (k *Key, err error) {
	appID := r.GetApp()
	namespace := r.GetNameSpace()
	for _, e := range r.Pathelement {
		k = &Key{
			kind:      e.GetType(),
			stringID:  e.GetName(),
			intID:     e.GetId(),
			parent:    k,
			appID:     appID,
			namespace: namespace,
		}
		if !k.valid() {
			return nil, ErrInvalidKey
		}
	}
	return
}

// keyToReferenceValue is the same as keyToProto except the output is a
// PropertyValue_ReferenceValue instead of a Reference.
func keyToReferenceValue(defaultAppID string, k *Key) *pb.PropertyValue_ReferenceValue {
	ref := keyToProto(defaultAppID, k)
	pe := make([]*pb.PropertyValue_ReferenceValue_PathElement, len(ref.Path.Element))
	for i, e := range ref.Path.Element {
		pe[i] = &pb.PropertyValue_ReferenceValue_PathElement{
			Type: e.Type,
			Id:   e.Id,
			Name: e.Name,
		}
	}
	return &pb.PropertyValue_ReferenceValue{
		App:         ref.App,
		NameSpace:   ref.NameSpace,
		Pathelement: pe,
	}
}

type multiArgType int

const (
	multiArgTypeInvalid multiArgType = iota
	multiArgTypePropertyLoadSaver
	multiArgTypeStruct
	multiArgTypeStructPtr
	multiArgTypeInterface
)

// checkMultiArg checks that v has type []S, []*S, []I, or []P, for some struct
// type S, for some interface type I, or some non-interface non-pointer type P
// such that P or *P implements PropertyLoadSaver.
//
// It returns what category the slice's elements are, and the reflect.Type
// that represents S, I or P.
//
// As a special case, PropertyList is an invalid type for v.
func checkMultiArg(v reflect.Value) (m multiArgType, elemType reflect.Type) {
	if v.Kind() != reflect.Slice {
		return multiArgTypeInvalid, nil
	}
	if v.Type() == typeOfPropertyList {
		return multiArgTypeInvalid, nil
	}
	elemType = v.Type().Elem()
	if reflect.PtrTo(elemType).Implements(typeOfPropertyLoadSaver) {
		return multiArgTypePropertyLoadSaver, elemType
	}
	switch elemType.Kind() {
	case reflect.Struct:
		return multiArgTypeStruct, elemType
	case reflect.Interface:
		return multiArgTypeInterface, elemType
	case reflect.Ptr:
		elemType = elemType.Elem()
		if elemType.Kind() == reflect.Struct {
			return multiArgTypeStructPtr, elemType
		}
	}
	return multiArgTypeInvalid, nil
}

// Get loads the entity stored for k into dst, which must be a struct pointer
// or implement PropertyLoadSaver. If there is no such entity for the key, Get
// returns ErrNoSuchEntity.
//
// The values of dst's unmatched struct fields are not modified, and matching
// slice-typed fields are not reset before appending to them. In particular, it
// is recommended to pass a pointer to a zero valued struct on each Get call.
//
// ErrFieldMismatch is returned when a field is to be loaded into a different
// type than the one it was stored from, or when a field is missing or
// unexported in the destination struct. ErrFieldMismatch is only returned if
// dst is a struct pointer.
func Get(c context.Context, key *Key, dst interface{}) error {
	if dst == nil { // GetMulti catches nil interface; we need to catch nil ptr here
		return ErrInvalidEntityType
	}
	err := GetMulti(c, []*Key{key}, []interface{}{dst})
	if me, ok := err.(appengine.MultiError); ok {
		return me[0]
	}
	return err
}

// GetMulti is a batch version of Get.
//
// dst must be a []S, []*S, []I or []P, for some struct type S, some interface
// type I, or some non-interface non-pointer type P such that P or *P
// implements PropertyLoadSaver. If an []I, each element must be a valid dst
// for Get: it must be a struct pointer or implement PropertyLoadSaver.
//
// As a special case, PropertyList is an invalid type for dst, even though a
// PropertyList is a slice of structs. It is treated as invalid to avoid being
// mistakenly passed when []PropertyList was intended.
func GetMulti(c context.Context, key []*Key, dst interface{}) error {
	v := reflect.ValueOf(dst)
	multiArgType, _ := checkMultiArg(v)
	if multiArgType == multiArgTypeInvalid {
		return errors.New("datastore: dst has invalid type")
	}
	if len(key) != v.Len() {
		return errors.New("datastore: key and dst slices have different length")
	}
	if len(key) == 0 {
		return nil
	}
	if err := multiValid(key); err != nil {
		return err
	}
	req := &pb.GetRequest{
		Key: multiKeyToProto(internal.FullyQualifiedAppID(c), key),
	}
	res := &pb.GetResponse{}
	if err := internal.Call(c, "datastore_v3", "Get", req, res); err != nil {
		return err
	}
	if len(key) != len(res.Entity) {
		return errors.New("datastore: internal error: server returned the wrong number of entities")
	}
	multiErr, any := make(appengine.MultiError, len(key)), false
	for i, e := range res.Entity {
		if e.Entity == nil {
			multiErr[i] = ErrNoSuchEntity
		} else {
			elem := v.Index(i)
			if multiArgType == multiArgTypePropertyLoadSaver || multiArgType == multiArgTypeStruct {
				elem = elem.Addr()
			}
			if multiArgType == multiArgTypeStructPtr && elem.IsNil() {
				elem.Set(reflect.New(elem.Type().Elem()))
			}
			multiErr[i] = loadEntity(elem.Interface(), e.Entity)
		}
		if multiErr[i] != nil {
			any = true
		}
	}
	if any {
		return multiErr
	}
	return nil
}

// Put saves the entity src into the datastore with key k. src must be a struct
// pointer or implement PropertyLoadSaver; if a struct pointer then any
// unexported fields of that struct will be skipped. If k is an incomplete key,
// the returned key will be a unique key generated by the datastore.
func Put(c context.Context, key *Key, src interface{}) (*Key, error) {
	k, err := PutMulti(c, []*Key{key}, []interface{}{src})
	if err != nil {
		if me, ok := err.(appengine.MultiError); ok {
			return nil, me[0]
		}
		return nil, err
	}
	return k[0], nil
}

// PutMulti is a batch version of Put.
//
// src must satisfy the same conditions as the dst argument to GetMulti.
func PutMulti(c context.Context, key []*Key, src interface{}) ([]*Key, error) {
	v := reflect.ValueOf(src)
	multiArgType, _ := checkMultiArg(v)
	if multiArgType == multiArgTypeInvalid {
		return nil, errors.New("datastore: src has invalid type")
	}
	if len(key) != v.Len() {
		return nil, errors.New("datastore: key and src slices have different length")
	}
	if len(key) == 0 {
		return nil, nil
	}
	appID := internal.FullyQualifiedAppID(c)
	if err := multiValid(key); err != nil {
		return nil, err
	}
	req := &pb.PutRequest{}
	for i := range key {
		elem := v.Index(i)
		if multiArgType == multiArgTypePropertyLoadSaver || multiArgType == multiArgTypeStruct {
			elem = elem.Addr()
		}
		sProto, err := saveEntity(appID, key[i], elem.Interface())
		if err != nil {
			return nil, err
		}
		req.Entity = append(req.Entity, sProto)
	}
	res := &pb.PutResponse{}
	if err := internal.Call(c, "datastore_v3", "Put", req, res); err != nil {
		return nil, err
	}
	if len(key) != len(res.Key) {
		return nil, errors.New("datastore: internal error: server returned the wrong number of keys")
	}
	ret := make([]*Key, len(key))
	for i := range ret {
		var err error
		ret[i], err = protoToKey(res.Key[i])
		if err != nil || ret[i].Incomplete() {
			return nil, errors.New("datastore: internal error: server returned an invalid key")
		}
	}
	return ret, nil
}

// Delete deletes the entity for the given key.
func Delete(c context.Context, key *Key) error {
	err := DeleteMulti(c, []*Key{key})
	if me, ok := err.(appengine.MultiError); ok {
		return me[0]
	}
	return err
}

// DeleteMulti is a batch version of Delete.
func DeleteMulti(c context.Context, key []*Key) error {
	if len(key) == 0 {
		return nil
	}
	if err := multiValid(key); err != nil {
		return err
	}
	req := &pb.DeleteRequest{
		Key: multiKeyToProto(internal.FullyQualifiedAppID(c), key),
	}
	res := &pb.DeleteResponse{}
	return internal.Call(c, "datastore_v3", "Delete", req, res)
}

func namespaceMod(m proto.Message, namespace string) {
	// pb.Query is the only type that has a name_space field.
	// All other namespace support in datastore is in the keys.
	switch m := m.(type) {
	case *pb.Query:
		if m.NameSpace == nil {
			m.NameSpace = &namespace
		}
	}
}

func init() {
	internal.NamespaceMods["datastore_v3"] = namespaceMod
	internal.RegisterErrorCodeMap("datastore_v3", pb.Error_ErrorCode_name)
	internal.RegisterTimeoutErrorCode("datastore_v3", int32(pb.Error_TIMEOUT))
}
