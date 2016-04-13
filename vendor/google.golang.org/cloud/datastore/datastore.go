// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package datastore contains a Google Cloud Datastore client.
//
// This package is experimental and may make backwards-incompatible changes.
package datastore // import "google.golang.org/cloud/datastore"

import (
	"errors"
	"fmt"
	"os"
	"reflect"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/cloud"
	pb "google.golang.org/cloud/internal/datastore"
	"google.golang.org/cloud/internal/transport"
)

const (
	prodHost = "https://www.googleapis.com"
	baseURL  = "/datastore/v1beta2/datasets/"

	userAgent = "gcloud-golang-datastore/20150727"
)

const (
	// ScopeDatastore grants permissions to view and/or manage datastore entities
	ScopeDatastore = "https://www.googleapis.com/auth/datastore"

	// ScopeUserEmail grants permission to view the user's email address.
	// It is required to access the datastore.
	ScopeUserEmail = "https://www.googleapis.com/auth/userinfo.email"
)

// protoClient is an interface for *transport.ProtoClient to support injecting
// fake clients in tests.
type protoClient interface {
	Call(context.Context, string, proto.Message, proto.Message) error
}

// Client is a client for reading and writing data in a datastore dataset.
type Client struct {
	client   protoClient
	endpoint string
	dataset  string // Called dataset by the datastore API, synonym for project ID.
}

// NewClient creates a new Client for a given dataset.
// If the project ID is empty, it is derived from the DATASTORE_DATASET environment variable.
func NewClient(ctx context.Context, projectID string, opts ...cloud.ClientOption) (*Client, error) {
	// Environment variables for gcd and gcloud emulator:
	// https://cloud.google.com/datastore/docs/tools/
	// https://cloud.google.com/sdk/gcloud/reference/beta/emulators/datastore/
	host := os.Getenv("DATASTORE_HOST")
	if host == "" {
		host = prodHost
	}
	if projectID == "" {
		projectID = os.Getenv("DATASTORE_DATASET")
	}
	if projectID == "" {
		return nil, errors.New("datastore: missing project/dataset id")
	}
	o := []cloud.ClientOption{
		cloud.WithEndpoint(host + baseURL),
		cloud.WithScopes(ScopeDatastore, ScopeUserEmail),
		cloud.WithUserAgent(userAgent),
	}
	o = append(o, opts...)
	client, err := transport.NewProtoClient(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &Client{
		client:  client,
		dataset: projectID,
	}, nil

}

var (
	// ErrInvalidEntityType is returned when functions like Get or Next are
	// passed a dst or src argument of invalid type.
	ErrInvalidEntityType = errors.New("datastore: invalid entity type")
	// ErrInvalidKey is returned when an invalid key is presented.
	ErrInvalidKey = errors.New("datastore: invalid key")
	// ErrNoSuchEntity is returned when no entity was found for a given key.
	ErrNoSuchEntity = errors.New("datastore: no such entity")
)

type multiArgType int

const (
	multiArgTypeInvalid multiArgType = iota
	multiArgTypePropertyLoadSaver
	multiArgTypeStruct
	multiArgTypeStructPtr
	multiArgTypeInterface
)

// nsKey is the type of the context.Context key to store the datastore
// namespace.
type nsKey struct{}

// WithNamespace returns a new context that limits the scope its parent
// context with a Datastore namespace.
func WithNamespace(parent context.Context, namespace string) context.Context {
	return context.WithValue(parent, nsKey{}, namespace)
}

// ctxNamespace returns the active namespace for a context.
// It defaults to "" if no namespace was specified.
func ctxNamespace(ctx context.Context) string {
	v, _ := ctx.Value(nsKey{}).(string)
	return v
}

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

func (c *Client) call(ctx context.Context, method string, req, resp proto.Message) error {
	return c.client.Call(ctx, c.dataset+"/"+method, req, resp)
}

func keyToProto(k *Key) *pb.Key {
	if k == nil {
		return nil
	}

	// TODO(jbd): Eliminate unrequired allocations.
	path := []*pb.Key_PathElement(nil)
	for {
		el := &pb.Key_PathElement{
			Kind: proto.String(k.kind),
		}
		if k.id != 0 {
			el.Id = proto.Int64(k.id)
		}
		if k.name != "" {
			el.Name = proto.String(k.name)
		}
		path = append([]*pb.Key_PathElement{el}, path...)
		if k.parent == nil {
			break
		}
		k = k.parent
	}
	key := &pb.Key{
		PathElement: path,
	}
	if k.namespace != "" {
		key.PartitionId = &pb.PartitionId{
			Namespace: proto.String(k.namespace),
		}
	}
	return key
}

// protoToKey decodes a protocol buffer representation of a key into an
// equivalent *Key object.
func protoToKey(p *pb.Key) (*Key, error) {
	var key *Key
	for _, el := range p.GetPathElement() {
		key = &Key{
			namespace: p.GetPartitionId().GetNamespace(),
			kind:      el.GetKind(),
			id:        el.GetId(),
			name:      el.GetName(),
			parent:    key,
		}
	}
	if !key.valid() { // Also detects key == nil.
		return nil, ErrInvalidKey
	}
	return key, nil
}

// multiKeyToProto is a batch version of keyToProto.
func multiKeyToProto(keys []*Key) []*pb.Key {
	ret := make([]*pb.Key, len(keys))
	for i, k := range keys {
		ret[i] = keyToProto(k)
	}
	return ret
}

// multiKeyToProto is a batch version of keyToProto.
func multiProtoToKey(keys []*pb.Key) ([]*Key, error) {
	hasErr := false
	ret := make([]*Key, len(keys))
	err := make(MultiError, len(keys))
	for i, k := range keys {
		ret[i], err[i] = protoToKey(k)
		if err[i] != nil {
			hasErr = true
		}
	}
	if hasErr {
		return nil, err
	}
	return ret, nil
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
	err := make(MultiError, len(key))
	for i, k := range key {
		if !k.valid() {
			err[i] = ErrInvalidKey
		}
	}
	return err
}

// checkMultiArg checks that v has type []S, []*S, []I, or []P, for some struct
// type S, for some interface type I, or some non-interface non-pointer type P
// such that P or *P implements PropertyLoadSaver.
//
// It returns what category the slice's elements are, and the reflect.Type
// that represents S, I or P.
//
// As a special case, PropertyList is an invalid type for v.
//
// TODO(djd): multiArg is very confusing. Fold this logic into the
// relevant Put/Get methods to make the logic less opaque.
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

// Get loads the entity stored for key into dst, which must be a struct pointer
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
func (c *Client) Get(ctx context.Context, key *Key, dst interface{}) error {
	if dst == nil { // get catches nil interfaces; we need to catch nil ptr here
		return ErrInvalidEntityType
	}
	err := c.get(ctx, []*Key{key}, []interface{}{dst}, nil)
	if me, ok := err.(MultiError); ok {
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
func (c *Client) GetMulti(ctx context.Context, keys []*Key, dst interface{}) error {
	return c.get(ctx, keys, dst, nil)
}

func (c *Client) get(ctx context.Context, keys []*Key, dst interface{}, opts *pb.ReadOptions) error {
	v := reflect.ValueOf(dst)
	multiArgType, _ := checkMultiArg(v)

	// Sanity checks
	if multiArgType == multiArgTypeInvalid {
		return errors.New("datastore: dst has invalid type")
	}
	if len(keys) != v.Len() {
		return errors.New("datastore: keys and dst slices have different length")
	}
	if len(keys) == 0 {
		return nil
	}

	// Go through keys, validate them, serialize then, and create a dict mapping them to their index
	multiErr, any := make(MultiError, len(keys)), false
	keyMap := make(map[string]int)
	pbKeys := make([]*pb.Key, len(keys))
	for i, k := range keys {
		if !k.valid() {
			multiErr[i] = ErrInvalidKey
			any = true
		} else {
			keyMap[k.String()] = i
			pbKeys[i] = keyToProto(k)
		}
	}
	if any {
		return multiErr
	}
	req := &pb.LookupRequest{
		Key:         pbKeys,
		ReadOptions: opts,
	}
	resp := &pb.LookupResponse{}
	if err := c.call(ctx, "lookup", req, resp); err != nil {
		return err
	}
	if len(resp.Deferred) > 0 {
		// TODO(jbd): Assess whether we should retry the deferred keys.
		return errors.New("datastore: some entities temporarily unavailable")
	}
	if len(keys) != len(resp.Found)+len(resp.Missing) {
		return errors.New("datastore: internal error: server returned the wrong number of entities")
	}
	for _, e := range resp.Found {
		k, err := protoToKey(e.Entity.Key)
		if err != nil {
			return errors.New("datastore: internal error: server returned an invalid key")
		}
		index := keyMap[k.String()]
		elem := v.Index(index)
		if multiArgType == multiArgTypePropertyLoadSaver || multiArgType == multiArgTypeStruct {
			elem = elem.Addr()
		}
		if multiArgType == multiArgTypeStructPtr && elem.IsNil() {
			elem.Set(reflect.New(elem.Type().Elem()))
		}
		if err := loadEntity(elem.Interface(), e.Entity); err != nil {
			multiErr[index] = err
			any = true
		}
	}
	for _, e := range resp.Missing {
		k, err := protoToKey(e.Entity.Key)
		if err != nil {
			return errors.New("datastore: internal error: server returned an invalid key")
		}
		multiErr[keyMap[k.String()]] = ErrNoSuchEntity
		any = true
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
func (c *Client) Put(ctx context.Context, key *Key, src interface{}) (*Key, error) {
	k, err := c.PutMulti(ctx, []*Key{key}, []interface{}{src})
	if err != nil {
		if me, ok := err.(MultiError); ok {
			return nil, me[0]
		}
		return nil, err
	}
	return k[0], nil
}

// PutMulti is a batch version of Put.
//
// src must satisfy the same conditions as the dst argument to GetMulti.
func (c *Client) PutMulti(ctx context.Context, keys []*Key, src interface{}) ([]*Key, error) {
	mutation, err := putMutation(keys, src)
	if err != nil {
		return nil, err
	}

	// Make the request.
	req := &pb.CommitRequest{
		Mutation: mutation,
		Mode:     pb.CommitRequest_NON_TRANSACTIONAL.Enum(),
	}
	resp := &pb.CommitResponse{}
	if err := c.call(ctx, "commit", req, resp); err != nil {
		return nil, err
	}

	// Copy any newly minted keys into the returned keys.
	newKeys := make(map[int]int) // Map of index in returned slice to index in response.
	ret := make([]*Key, len(keys))
	var idx int
	for i, key := range keys {
		if key.Incomplete() {
			// This key will be in the mutation result.
			newKeys[i] = idx
			idx++
		} else {
			ret[i] = key
		}
	}
	if len(newKeys) != len(resp.MutationResult.InsertAutoIdKey) {
		return nil, errors.New("datastore: internal error: server returned the wrong number of keys")
	}
	for retI, respI := range newKeys {
		ret[retI], err = protoToKey(resp.MutationResult.InsertAutoIdKey[respI])
		if err != nil {
			return nil, errors.New("datastore: internal error: server returned an invalid key")
		}
	}
	return ret, nil
}

func putMutation(keys []*Key, src interface{}) (*pb.Mutation, error) {
	v := reflect.ValueOf(src)
	multiArgType, _ := checkMultiArg(v)
	if multiArgType == multiArgTypeInvalid {
		return nil, errors.New("datastore: src has invalid type")
	}
	if len(keys) != v.Len() {
		return nil, errors.New("datastore: key and src slices have different length")
	}
	if len(keys) == 0 {
		return nil, nil
	}
	if err := multiValid(keys); err != nil {
		return nil, err
	}
	var upsert, insert []*pb.Entity
	for i, k := range keys {
		elem := v.Index(i)
		// Two cases where we need to take the address:
		// 1) multiArgTypePropertyLoadSaver => &elem implements PLS
		// 2) multiArgTypeStruct => saveEntity needs *struct
		if multiArgType == multiArgTypePropertyLoadSaver || multiArgType == multiArgTypeStruct {
			elem = elem.Addr()
		}
		p, err := saveEntity(k, elem.Interface())
		if err != nil {
			return nil, fmt.Errorf("datastore: Error while saving %v: %v", k.String(), err)
		}
		if k.Incomplete() {
			insert = append(insert, p)
		} else {
			upsert = append(upsert, p)
		}
	}

	return &pb.Mutation{
		InsertAutoId: insert,
		Upsert:       upsert,
	}, nil
}

// Delete deletes the entity for the given key.
func (c *Client) Delete(ctx context.Context, key *Key) error {
	err := c.DeleteMulti(ctx, []*Key{key})
	if me, ok := err.(MultiError); ok {
		return me[0]
	}
	return err
}

// DeleteMulti is a batch version of Delete.
func (c *Client) DeleteMulti(ctx context.Context, keys []*Key) error {
	mutation, err := deleteMutation(keys)
	if err != nil {
		return err
	}

	req := &pb.CommitRequest{
		Mutation: mutation,
		Mode:     pb.CommitRequest_NON_TRANSACTIONAL.Enum(),
	}
	resp := &pb.CommitResponse{}
	return c.call(ctx, "commit", req, resp)
}

func deleteMutation(keys []*Key) (*pb.Mutation, error) {
	protoKeys := make([]*pb.Key, len(keys))
	for i, k := range keys {
		if k.Incomplete() {
			return nil, fmt.Errorf("datastore: can't delete the incomplete key: %v", k)
		}
		protoKeys[i] = keyToProto(k)
	}

	return &pb.Mutation{
		Delete: protoKeys,
	}, nil
}
