// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package memcache provides a client for App Engine's distributed in-memory
// key-value store for small chunks of arbitrary data.
//
// The fundamental operations get and set items, keyed by a string.
//
//	item0, err := memcache.Get(c, "key")
//	if err != nil && err != memcache.ErrCacheMiss {
//		return err
//	}
//	if err == nil {
//		fmt.Fprintf(w, "memcache hit: Key=%q Val=[% x]\n", item0.Key, item0.Value)
//	} else {
//		fmt.Fprintf(w, "memcache miss\n")
//	}
//
// and
//
//	item1 := &memcache.Item{
//		Key:   "foo",
//		Value: []byte("bar"),
//	}
//	if err := memcache.Set(c, item1); err != nil {
//		return err
//	}
package memcache // import "google.golang.org/appengine/memcache"

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/memcache"
)

var (
	// ErrCacheMiss means that an operation failed
	// because the item wasn't present.
	ErrCacheMiss = errors.New("memcache: cache miss")
	// ErrCASConflict means that a CompareAndSwap call failed due to the
	// cached value being modified between the Get and the CompareAndSwap.
	// If the cached value was simply evicted rather than replaced,
	// ErrNotStored will be returned instead.
	ErrCASConflict = errors.New("memcache: compare-and-swap conflict")
	// ErrNoStats means that no statistics were available.
	ErrNoStats = errors.New("memcache: no statistics available")
	// ErrNotStored means that a conditional write operation (i.e. Add or
	// CompareAndSwap) failed because the condition was not satisfied.
	ErrNotStored = errors.New("memcache: item not stored")
	// ErrServerError means that a server error occurred.
	ErrServerError = errors.New("memcache: server error")
)

// Item is the unit of memcache gets and sets.
type Item struct {
	// Key is the Item's key (250 bytes maximum).
	Key string
	// Value is the Item's value.
	Value []byte
	// Object is the Item's value for use with a Codec.
	Object interface{}
	// Flags are server-opaque flags whose semantics are entirely up to the
	// App Engine app.
	Flags uint32
	// Expiration is the maximum duration that the item will stay
	// in the cache.
	// The zero value means the Item has no expiration time.
	// Subsecond precision is ignored.
	// This is not set when getting items.
	Expiration time.Duration
	// casID is a client-opaque value used for compare-and-swap operations.
	// Zero means that compare-and-swap is not used.
	casID uint64
}

const (
	secondsIn30Years = 60 * 60 * 24 * 365 * 30 // from memcache server code
	thirtyYears      = time.Duration(secondsIn30Years) * time.Second
)

// protoToItem converts a protocol buffer item to a Go struct.
func protoToItem(p *pb.MemcacheGetResponse_Item) *Item {
	return &Item{
		Key:   string(p.Key),
		Value: p.Value,
		Flags: p.GetFlags(),
		casID: p.GetCasId(),
	}
}

// If err is an appengine.MultiError, return its first element. Otherwise, return err.
func singleError(err error) error {
	if me, ok := err.(appengine.MultiError); ok {
		return me[0]
	}
	return err
}

// Get gets the item for the given key. ErrCacheMiss is returned for a memcache
// cache miss. The key must be at most 250 bytes in length.
func Get(c context.Context, key string) (*Item, error) {
	m, err := GetMulti(c, []string{key})
	if err != nil {
		return nil, err
	}
	if _, ok := m[key]; !ok {
		return nil, ErrCacheMiss
	}
	return m[key], nil
}

// GetMulti is a batch version of Get. The returned map from keys to items may
// have fewer elements than the input slice, due to memcache cache misses.
// Each key must be at most 250 bytes in length.
func GetMulti(c context.Context, key []string) (map[string]*Item, error) {
	if len(key) == 0 {
		return nil, nil
	}
	keyAsBytes := make([][]byte, len(key))
	for i, k := range key {
		keyAsBytes[i] = []byte(k)
	}
	req := &pb.MemcacheGetRequest{
		Key:    keyAsBytes,
		ForCas: proto.Bool(true),
	}
	res := &pb.MemcacheGetResponse{}
	if err := internal.Call(c, "memcache", "Get", req, res); err != nil {
		return nil, err
	}
	m := make(map[string]*Item, len(res.Item))
	for _, p := range res.Item {
		t := protoToItem(p)
		m[t.Key] = t
	}
	return m, nil
}

// Delete deletes the item for the given key.
// ErrCacheMiss is returned if the specified item can not be found.
// The key must be at most 250 bytes in length.
func Delete(c context.Context, key string) error {
	return singleError(DeleteMulti(c, []string{key}))
}

// DeleteMulti is a batch version of Delete.
// If any keys cannot be found, an appengine.MultiError is returned.
// Each key must be at most 250 bytes in length.
func DeleteMulti(c context.Context, key []string) error {
	if len(key) == 0 {
		return nil
	}
	req := &pb.MemcacheDeleteRequest{
		Item: make([]*pb.MemcacheDeleteRequest_Item, len(key)),
	}
	for i, k := range key {
		req.Item[i] = &pb.MemcacheDeleteRequest_Item{Key: []byte(k)}
	}
	res := &pb.MemcacheDeleteResponse{}
	if err := internal.Call(c, "memcache", "Delete", req, res); err != nil {
		return err
	}
	if len(res.DeleteStatus) != len(key) {
		return ErrServerError
	}
	me, any := make(appengine.MultiError, len(key)), false
	for i, s := range res.DeleteStatus {
		switch s {
		case pb.MemcacheDeleteResponse_DELETED:
			// OK
		case pb.MemcacheDeleteResponse_NOT_FOUND:
			me[i] = ErrCacheMiss
			any = true
		default:
			me[i] = ErrServerError
			any = true
		}
	}
	if any {
		return me
	}
	return nil
}

// Increment atomically increments the decimal value in the given key
// by delta and returns the new value. The value must fit in a uint64.
// Overflow wraps around, and underflow is capped to zero. The
// provided delta may be negative. If the key doesn't exist in
// memcache, the provided initial value is used to atomically
// populate it before the delta is applied.
// The key must be at most 250 bytes in length.
func Increment(c context.Context, key string, delta int64, initialValue uint64) (newValue uint64, err error) {
	return incr(c, key, delta, &initialValue)
}

// IncrementExisting works like Increment but assumes that the key
// already exists in memcache and doesn't take an initial value.
// IncrementExisting can save work if calculating the initial value is
// expensive.
// An error is returned if the specified item can not be found.
func IncrementExisting(c context.Context, key string, delta int64) (newValue uint64, err error) {
	return incr(c, key, delta, nil)
}

func incr(c context.Context, key string, delta int64, initialValue *uint64) (newValue uint64, err error) {
	req := &pb.MemcacheIncrementRequest{
		Key:          []byte(key),
		InitialValue: initialValue,
	}
	if delta >= 0 {
		req.Delta = proto.Uint64(uint64(delta))
	} else {
		req.Delta = proto.Uint64(uint64(-delta))
		req.Direction = pb.MemcacheIncrementRequest_DECREMENT.Enum()
	}
	res := &pb.MemcacheIncrementResponse{}
	err = internal.Call(c, "memcache", "Increment", req, res)
	if err != nil {
		return
	}
	if res.NewValue == nil {
		return 0, ErrCacheMiss
	}
	return *res.NewValue, nil
}

// set sets the given items using the given conflict resolution policy.
// appengine.MultiError may be returned.
func set(c context.Context, item []*Item, value [][]byte, policy pb.MemcacheSetRequest_SetPolicy) error {
	if len(item) == 0 {
		return nil
	}
	req := &pb.MemcacheSetRequest{
		Item: make([]*pb.MemcacheSetRequest_Item, len(item)),
	}
	for i, t := range item {
		p := &pb.MemcacheSetRequest_Item{
			Key: []byte(t.Key),
		}
		if value == nil {
			p.Value = t.Value
		} else {
			p.Value = value[i]
		}
		if t.Flags != 0 {
			p.Flags = proto.Uint32(t.Flags)
		}
		if t.Expiration != 0 {
			// In the .proto file, MemcacheSetRequest_Item uses a fixed32 (i.e. unsigned)
			// for expiration time, while MemcacheGetRequest_Item uses int32 (i.e. signed).
			// Throughout this .go file, we use int32.
			// Also, in the proto, the expiration value is either a duration (in seconds)
			// or an absolute Unix timestamp (in seconds), depending on whether the
			// value is less than or greater than or equal to 30 years, respectively.
			if t.Expiration < time.Second {
				// Because an Expiration of 0 means no expiration, we take
				// care here to translate an item with an expiration
				// Duration between 0-1 seconds as immediately expiring
				// (saying it expired a few seconds ago), rather than
				// rounding it down to 0 and making it live forever.
				p.ExpirationTime = proto.Uint32(uint32(time.Now().Unix()) - 5)
			} else if t.Expiration >= thirtyYears {
				p.ExpirationTime = proto.Uint32(uint32(time.Now().Unix()) + uint32(t.Expiration/time.Second))
			} else {
				p.ExpirationTime = proto.Uint32(uint32(t.Expiration / time.Second))
			}
		}
		if t.casID != 0 {
			p.CasId = proto.Uint64(t.casID)
			p.ForCas = proto.Bool(true)
		}
		p.SetPolicy = policy.Enum()
		req.Item[i] = p
	}
	res := &pb.MemcacheSetResponse{}
	if err := internal.Call(c, "memcache", "Set", req, res); err != nil {
		return err
	}
	if len(res.SetStatus) != len(item) {
		return ErrServerError
	}
	me, any := make(appengine.MultiError, len(item)), false
	for i, st := range res.SetStatus {
		var err error
		switch st {
		case pb.MemcacheSetResponse_STORED:
			// OK
		case pb.MemcacheSetResponse_NOT_STORED:
			err = ErrNotStored
		case pb.MemcacheSetResponse_EXISTS:
			err = ErrCASConflict
		default:
			err = ErrServerError
		}
		if err != nil {
			me[i] = err
			any = true
		}
	}
	if any {
		return me
	}
	return nil
}

// Set writes the given item, unconditionally.
func Set(c context.Context, item *Item) error {
	return singleError(set(c, []*Item{item}, nil, pb.MemcacheSetRequest_SET))
}

// SetMulti is a batch version of Set.
// appengine.MultiError may be returned.
func SetMulti(c context.Context, item []*Item) error {
	return set(c, item, nil, pb.MemcacheSetRequest_SET)
}

// Add writes the given item, if no value already exists for its key.
// ErrNotStored is returned if that condition is not met.
func Add(c context.Context, item *Item) error {
	return singleError(set(c, []*Item{item}, nil, pb.MemcacheSetRequest_ADD))
}

// AddMulti is a batch version of Add.
// appengine.MultiError may be returned.
func AddMulti(c context.Context, item []*Item) error {
	return set(c, item, nil, pb.MemcacheSetRequest_ADD)
}

// CompareAndSwap writes the given item that was previously returned by Get,
// if the value was neither modified or evicted between the Get and the
// CompareAndSwap calls. The item's Key should not change between calls but
// all other item fields may differ.
// ErrCASConflict is returned if the value was modified in between the calls.
// ErrNotStored is returned if the value was evicted in between the calls.
func CompareAndSwap(c context.Context, item *Item) error {
	return singleError(set(c, []*Item{item}, nil, pb.MemcacheSetRequest_CAS))
}

// CompareAndSwapMulti is a batch version of CompareAndSwap.
// appengine.MultiError may be returned.
func CompareAndSwapMulti(c context.Context, item []*Item) error {
	return set(c, item, nil, pb.MemcacheSetRequest_CAS)
}

// Codec represents a symmetric pair of functions that implement a codec.
// Items stored into or retrieved from memcache using a Codec have their
// values marshaled or unmarshaled.
//
// All the methods provided for Codec behave analogously to the package level
// function with same name.
type Codec struct {
	Marshal   func(interface{}) ([]byte, error)
	Unmarshal func([]byte, interface{}) error
}

// Get gets the item for the given key and decodes the obtained value into v.
// ErrCacheMiss is returned for a memcache cache miss.
// The key must be at most 250 bytes in length.
func (cd Codec) Get(c context.Context, key string, v interface{}) (*Item, error) {
	i, err := Get(c, key)
	if err != nil {
		return nil, err
	}
	if err := cd.Unmarshal(i.Value, v); err != nil {
		return nil, err
	}
	return i, nil
}

func (cd Codec) set(c context.Context, items []*Item, policy pb.MemcacheSetRequest_SetPolicy) error {
	var vs [][]byte
	var me appengine.MultiError
	for i, item := range items {
		v, err := cd.Marshal(item.Object)
		if err != nil {
			if me == nil {
				me = make(appengine.MultiError, len(items))
			}
			me[i] = err
			continue
		}
		if me == nil {
			vs = append(vs, v)
		}
	}
	if me != nil {
		return me
	}

	return set(c, items, vs, policy)
}

// Set writes the given item, unconditionally.
func (cd Codec) Set(c context.Context, item *Item) error {
	return singleError(cd.set(c, []*Item{item}, pb.MemcacheSetRequest_SET))
}

// SetMulti is a batch version of Set.
// appengine.MultiError may be returned.
func (cd Codec) SetMulti(c context.Context, items []*Item) error {
	return cd.set(c, items, pb.MemcacheSetRequest_SET)
}

// Add writes the given item, if no value already exists for its key.
// ErrNotStored is returned if that condition is not met.
func (cd Codec) Add(c context.Context, item *Item) error {
	return singleError(cd.set(c, []*Item{item}, pb.MemcacheSetRequest_ADD))
}

// AddMulti is a batch version of Add.
// appengine.MultiError may be returned.
func (cd Codec) AddMulti(c context.Context, items []*Item) error {
	return cd.set(c, items, pb.MemcacheSetRequest_ADD)
}

// CompareAndSwap writes the given item that was previously returned by Get,
// if the value was neither modified or evicted between the Get and the
// CompareAndSwap calls. The item's Key should not change between calls but
// all other item fields may differ.
// ErrCASConflict is returned if the value was modified in between the calls.
// ErrNotStored is returned if the value was evicted in between the calls.
func (cd Codec) CompareAndSwap(c context.Context, item *Item) error {
	return singleError(cd.set(c, []*Item{item}, pb.MemcacheSetRequest_CAS))
}

// CompareAndSwapMulti is a batch version of CompareAndSwap.
// appengine.MultiError may be returned.
func (cd Codec) CompareAndSwapMulti(c context.Context, items []*Item) error {
	return cd.set(c, items, pb.MemcacheSetRequest_CAS)
}

var (
	// Gob is a Codec that uses the gob package.
	Gob = Codec{gobMarshal, gobUnmarshal}
	// JSON is a Codec that uses the json package.
	JSON = Codec{json.Marshal, json.Unmarshal}
)

func gobMarshal(v interface{}) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobUnmarshal(data []byte, v interface{}) error {
	return gob.NewDecoder(bytes.NewBuffer(data)).Decode(v)
}

// Statistics represents a set of statistics about the memcache cache.
// This may include items that have expired but have not yet been removed from the cache.
type Statistics struct {
	Hits     uint64 // Counter of cache hits
	Misses   uint64 // Counter of cache misses
	ByteHits uint64 // Counter of bytes transferred for gets

	Items uint64 // Items currently in the cache
	Bytes uint64 // Size of all items currently in the cache

	Oldest int64 // Age of access of the oldest item, in seconds
}

// Stats retrieves the current memcache statistics.
func Stats(c context.Context) (*Statistics, error) {
	req := &pb.MemcacheStatsRequest{}
	res := &pb.MemcacheStatsResponse{}
	if err := internal.Call(c, "memcache", "Stats", req, res); err != nil {
		return nil, err
	}
	if res.Stats == nil {
		return nil, ErrNoStats
	}
	return &Statistics{
		Hits:     *res.Stats.Hits,
		Misses:   *res.Stats.Misses,
		ByteHits: *res.Stats.ByteHits,
		Items:    *res.Stats.Items,
		Bytes:    *res.Stats.Bytes,
		Oldest:   int64(*res.Stats.OldestItemAge),
	}, nil
}

// Flush flushes all items from memcache.
func Flush(c context.Context) error {
	req := &pb.MemcacheFlushRequest{}
	res := &pb.MemcacheFlushResponse{}
	return internal.Call(c, "memcache", "FlushAll", req, res)
}

func namespaceMod(m proto.Message, namespace string) {
	switch m := m.(type) {
	case *pb.MemcacheDeleteRequest:
		if m.NameSpace == nil {
			m.NameSpace = &namespace
		}
	case *pb.MemcacheGetRequest:
		if m.NameSpace == nil {
			m.NameSpace = &namespace
		}
	case *pb.MemcacheIncrementRequest:
		if m.NameSpace == nil {
			m.NameSpace = &namespace
		}
	case *pb.MemcacheSetRequest:
		if m.NameSpace == nil {
			m.NameSpace = &namespace
		}
		// MemcacheFlushRequest, MemcacheStatsRequest do not apply namespace.
	}
}

func init() {
	internal.RegisterErrorCodeMap("memcache", pb.MemcacheServiceError_ErrorCode_name)
	internal.NamespaceMods["memcache"] = namespaceMod
}
