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

package datastore

import (
	"bytes"
	"encoding/base64"
	"encoding/gob"
	"errors"
	"strconv"
	"strings"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	pb "google.golang.org/genproto/googleapis/datastore/v1"
)

// Key represents the datastore key for a stored entity, and is immutable.
type Key struct {
	kind   string
	id     int64
	name   string
	parent *Key

	namespace string
}

func (k *Key) Kind() string {
	return k.kind
}

func (k *Key) ID() int64 {
	return k.id
}

func (k *Key) Name() string {
	return k.name
}

func (k *Key) Parent() *Key {
	return k.parent
}

func (k *Key) SetParent(v *Key) {
	if v.Incomplete() {
		panic("can't set an incomplete key as parent")
	}
	k.parent = v
}

func (k *Key) Namespace() string {
	return k.namespace
}

// Complete returns whether the key does not refer to a stored entity.
func (k *Key) Incomplete() bool {
	return k.name == "" && k.id == 0
}

// valid returns whether the key is valid.
func (k *Key) valid() bool {
	if k == nil {
		return false
	}
	for ; k != nil; k = k.parent {
		if k.kind == "" {
			return false
		}
		if k.name != "" && k.id != 0 {
			return false
		}
		if k.parent != nil {
			if k.parent.Incomplete() {
				return false
			}
			if k.parent.namespace != k.namespace {
				return false
			}
		}
	}
	return true
}

func (k *Key) Equal(o *Key) bool {
	for {
		if k == nil || o == nil {
			return k == o // if either is nil, both must be nil
		}
		if k.namespace != o.namespace || k.name != o.name || k.id != o.id || k.kind != o.kind {
			return false
		}
		if k.parent == nil && o.parent == nil {
			return true
		}
		k = k.parent
		o = o.parent
	}
}

// marshal marshals the key's string representation to the buffer.
func (k *Key) marshal(b *bytes.Buffer) {
	if k.parent != nil {
		k.parent.marshal(b)
	}
	b.WriteByte('/')
	b.WriteString(k.kind)
	b.WriteByte(',')
	if k.name != "" {
		b.WriteString(k.name)
	} else {
		b.WriteString(strconv.FormatInt(k.id, 10))
	}
}

// String returns a string representation of the key.
func (k *Key) String() string {
	if k == nil {
		return ""
	}
	b := bytes.NewBuffer(make([]byte, 0, 512))
	k.marshal(b)
	return b.String()
}

// Note: Fields not renamed compared to appengine gobKey struct
// This ensures gobs created by appengine can be read here, and vice/versa
type gobKey struct {
	Kind      string
	StringID  string
	IntID     int64
	Parent    *gobKey
	AppID     string
	Namespace string
}

func keyToGobKey(k *Key) *gobKey {
	if k == nil {
		return nil
	}
	return &gobKey{
		Kind:      k.kind,
		StringID:  k.name,
		IntID:     k.id,
		Parent:    keyToGobKey(k.parent),
		Namespace: k.namespace,
	}
}

func gobKeyToKey(gk *gobKey) *Key {
	if gk == nil {
		return nil
	}
	return &Key{
		kind:      gk.Kind,
		name:      gk.StringID,
		id:        gk.IntID,
		parent:    gobKeyToKey(gk.Parent),
		namespace: gk.Namespace,
	}
}

func (k *Key) GobEncode() ([]byte, error) {
	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(keyToGobKey(k)); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (k *Key) GobDecode(buf []byte) error {
	gk := new(gobKey)
	if err := gob.NewDecoder(bytes.NewBuffer(buf)).Decode(gk); err != nil {
		return err
	}
	*k = *gobKeyToKey(gk)
	return nil
}

func (k *Key) MarshalJSON() ([]byte, error) {
	return []byte(`"` + k.Encode() + `"`), nil
}

func (k *Key) UnmarshalJSON(buf []byte) error {
	if len(buf) < 2 || buf[0] != '"' || buf[len(buf)-1] != '"' {
		return errors.New("datastore: bad JSON key")
	}
	k2, err := DecodeKey(string(buf[1 : len(buf)-1]))
	if err != nil {
		return err
	}
	*k = *k2
	return nil
}

// Encode returns an opaque representation of the key
// suitable for use in HTML and URLs.
// This is compatible with the Python and Java runtimes.
func (k *Key) Encode() string {
	pKey := keyToProto(k)

	b, err := proto.Marshal(pKey)
	if err != nil {
		panic(err)
	}

	// Trailing padding is stripped.
	return strings.TrimRight(base64.URLEncoding.EncodeToString(b), "=")
}

// DecodeKey decodes a key from the opaque representation returned by Encode.
func DecodeKey(encoded string) (*Key, error) {
	// Re-add padding.
	if m := len(encoded) % 4; m != 0 {
		encoded += strings.Repeat("=", 4-m)
	}

	b, err := base64.URLEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}

	pKey := new(pb.Key)
	if err := proto.Unmarshal(b, pKey); err != nil {
		return nil, err
	}
	return protoToKey(pKey)
}

// NewIncompleteKey creates a new incomplete key.
// kind cannot be empty.
func NewIncompleteKey(ctx context.Context, kind string, parent *Key) *Key {
	return NewKey(ctx, kind, "", 0, parent)
}

// NewKey creates a new key.
// kind cannot be empty.
// At least one of name and id must be zero. If both are zero, the key returned
// is incomplete.
// parent must either be a complete key or nil.
func NewKey(ctx context.Context, kind, name string, id int64, parent *Key) *Key {
	return &Key{
		kind:      kind,
		name:      name,
		id:        id,
		parent:    parent,
		namespace: ctxNamespace(ctx),
	}
}

// AllocateIDs accepts a slice of incomplete keys and returns a
// slice of complete keys that are guaranteed to be valid in the datastore
func (c *Client) AllocateIDs(ctx context.Context, keys []*Key) ([]*Key, error) {
	if keys == nil {
		return nil, nil
	}

	req := &pb.AllocateIdsRequest{
		ProjectId: c.dataset,
		Keys:      multiKeyToProto(keys),
	}
	resp, err := c.client.AllocateIds(ctx, req)
	if err != nil {
		return nil, err
	}

	return multiProtoToKey(resp.Keys)
}
