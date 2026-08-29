//go:build linux && amd64 && !race

/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bytecache

import (
	"container/list"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"sync"
	"unsafe"

	"k8s.io/klog/v2"
)

const (
	// Sparse reservation per store; pages materialize only as written.
	defaultArenaBytes = 1 << 30
	// Heap budget for the hot LRU of materialized objects.
	defaultHotBytes = 16 << 20
)

// Enabled reports whether the experimental bytecache store is switched on.
// Values: "1"/"reloc" (native relocation codec), "proto", "gob".
func Enabled() bool {
	v := os.Getenv("KUBE_BYTECACHE")
	return v != "" && v != "0"
}

func kindFromEnv() codecKind {
	switch os.Getenv("KUBE_BYTECACHE") {
	case "1", "reloc":
		return kindReloc
	case "proto":
		return kindProto
	case "gob":
		return kindGob
	}
	return kindReloc
}

func hotBudget() int {
	if v := os.Getenv("KUBE_BYTECACHE_HOT_BYTES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultHotBytes
}

// Cache is a key->object store holding objects at rest in a file-backed mmap
// arena, with a bounded LRU of materialized heap copies in front and a
// passthrough mode for objects that cannot be encoded.
//
// Locking contract: Set, Delete, and Replace require the caller's write lock;
// Get, GetTransient, All, Keys, and Len require at least the caller's read
// lock. The LRU has its own internal lock so Gets may run concurrently.
type Cache struct {
	kind        codecKind
	maxHotBytes int
	// typeState memoizes the per-type round-trip self-check: the first
	// object of each type is encoded, decoded, and DeepEqual-verified.
	// Types the codec cannot faithfully round-trip are blacklisted to
	// passthrough (e.g. gob silently drops unexported fields).
	typeState map[reflect.Type]int8 // 0 unknown, 1 ok, 2 bad
	arena     *Arena
	codec     *codec
	degraded  bool // arena setup failed: passthrough everything
	index     map[string]ent

	lmu      sync.Mutex // guards hot/lru/hotBytes (mutated during read-locked Gets)
	hot      map[string]*list.Element
	lru      *list.List
	hotBytes int

	encoded, passed int
	warnOnce        sync.Once
}

type hotEnt struct {
	key  string
	obj  interface{}
	size int
}

func New() *Cache {
	return &Cache{
		kind:        kindFromEnv(),
		typeState:   map[reflect.Type]int8{},
		maxHotBytes: hotBudget(),
		index:       map[string]ent{},
		hot:         map[string]*list.Element{},
		lru:         list.New(),
	}
}

func (c *Cache) ensureArena() bool {
	if c.degraded {
		return false
	}
	if c.arena != nil {
		return true
	}
	dir := os.Getenv("KUBE_BYTECACHE_DIR")
	if dir == "" {
		dir = os.TempDir()
	}
	f, err := os.CreateTemp(dir, "bytecache-*.arena")
	if err != nil {
		c.degrade("arena file", err)
		return false
	}
	path := f.Name()
	_ = f.Close()
	a, err := NewArena(path, defaultArenaBytes)
	if err != nil {
		_ = os.Remove(path)
		c.degrade("arena mmap", err)
		return false
	}
	// The mapping keeps the file alive; unlink so it never outlives us.
	_ = os.Remove(path)
	c.arena = a
	c.codec = newCodec(a)
	klog.Background().V(2).Info("bytecache: arena created", "reserveBytes", defaultArenaBytes, "hotBudgetBytes", c.maxHotBytes)
	return true
}

func (c *Cache) degrade(what string, err error) {
	c.degraded = true
	c.warnOnce.Do(func() {
		klog.Background().Error(err, "bytecache: falling back to heap storage", "operation", what)
	})
}

// encode returns an arena entry, or a passthrough entry if obj cannot be
// (or should not be) encoded.
func (c *Cache) encode(obj interface{}) ent {
	rv := reflect.ValueOf(obj)
	if rv.Kind() != reflect.Pointer || rv.IsNil() || rv.Type().Elem().Kind() != reflect.Struct {
		c.passed++
		return ent{obj: obj}
	}
	if !c.ensureArena() {
		c.passed++
		return ent{obj: obj}
	}
	typ := rv.Type().Elem()
	if c.typeState[typ] == 2 {
		c.passed++
		return ent{obj: obj}
	}
	e, err := c.encodeKind(rv, obj)
	if err != nil {
		// Unsupported type shape, arena full, etc: store the object itself.
		c.passed++
		klog.Background().V(4).Info("bytecache: storing object on heap", "type", fmt.Sprintf("%T", obj), "reason", err)
		return ent{obj: obj}
	}
	if c.typeState[typ] == 0 {
		// First object of this type: verify the codec round-trips it
		// faithfully before trusting the encoding for the type.
		decoded, derr := c.decode(e)
		if derr != nil || !reflect.DeepEqual(obj, decoded) {
			c.typeState[typ] = 2
			klog.Background().Error(derr, "bytecache: codec cannot faithfully round-trip type; storing on heap", "type", typ.String())
			c.passed++
			return ent{obj: obj}
		}
		c.typeState[typ] = 1
	}
	c.encoded++
	return e
}

func (c *Cache) encodeKind(rv reflect.Value, obj interface{}) (ent, error) {
	switch c.kind {
	case kindProto:
		return c.encodeProto(obj)
	case kindGob:
		return c.encodeGob(obj)
	default:
		return c.codec.encode(rv.Type().Elem(), unsafe.Pointer(rv.Pointer()))
	}
}

func (c *Cache) decode(e ent) (interface{}, error) {
	switch c.kind {
	case kindProto:
		return c.decodeProto(e)
	case kindGob:
		return c.decodeGob(e)
	default:
		return c.codec.decode(e)
	}
}

// Set stores obj under key (caller holds write lock).
func (c *Cache) Set(key string, obj interface{}) {
	c.index[key] = c.encode(obj)
	c.dropHot(key) // any cached materialization is now stale
}

// Delete removes key (caller holds write lock).
func (c *Cache) Delete(key string) {
	delete(c.index, key)
	c.dropHot(key)
}

// Get returns the object for key, using and populating the hot LRU
// (caller holds at least read lock).
func (c *Cache) Get(key string) (interface{}, bool) {
	e, ok := c.index[key]
	if !ok {
		return nil, false
	}
	if e.typ == nil {
		return e.obj, true
	}
	c.lmu.Lock()
	if el, hit := c.hot[key]; hit {
		c.lru.MoveToFront(el)
		obj := el.Value.(*hotEnt).obj
		c.lmu.Unlock()
		return obj, true
	}
	c.lmu.Unlock()
	obj, err := c.decode(e)
	if err != nil {
		klog.Background().Error(err, "bytecache: decode failed", "key", key)
		return nil, false
	}
	c.lmu.Lock()
	if el, hit := c.hot[key]; hit { // raced with another reader; keep theirs
		c.lru.MoveToFront(el)
		obj = el.Value.(*hotEnt).obj
	} else {
		c.hot[key] = c.lru.PushFront(&hotEnt{key: key, obj: obj, size: e.size})
		c.hotBytes += e.size
		for c.hotBytes > c.maxHotBytes && c.lru.Len() > 1 {
			c.dropHotLocked(c.lru.Back())
		}
	}
	c.lmu.Unlock()
	return obj, true
}

// GetTransient returns the object without inserting it into the LRU — for
// bulk sweeps (List) and internal reads that shouldn't churn the hot set.
func (c *Cache) GetTransient(key string) (interface{}, bool) {
	e, ok := c.index[key]
	if !ok {
		return nil, false
	}
	if e.typ == nil {
		return e.obj, true
	}
	c.lmu.Lock()
	if el, hit := c.hot[key]; hit {
		obj := el.Value.(*hotEnt).obj
		c.lmu.Unlock()
		return obj, true
	}
	c.lmu.Unlock()
	obj, err := c.decode(e)
	if err != nil {
		klog.Background().Error(err, "bytecache: decode failed", "key", key)
		return nil, false
	}
	return obj, true
}

// All materializes every object (caller holds read lock).
func (c *Cache) All() []interface{} {
	out := make([]interface{}, 0, len(c.index))
	for key := range c.index {
		if obj, ok := c.GetTransient(key); ok {
			out = append(out, obj)
		}
	}
	return out
}

func (c *Cache) Keys() []string {
	out := make([]string, 0, len(c.index))
	for key := range c.index {
		out = append(out, key)
	}
	return out
}

func (c *Cache) Len() int { return len(c.index) }

// Replace swaps in a whole new content set (caller holds write lock). The
// arena is reset first, so every relist doubles as compaction — safe because
// consumers only ever hold detached heap copies, never arena pointers.
func (c *Cache) Replace(items map[string]interface{}) {
	c.lmu.Lock()
	c.hot = map[string]*list.Element{}
	c.lru.Init()
	c.hotBytes = 0
	c.lmu.Unlock()
	if c.arena != nil {
		c.arena.Reset()
	}
	c.index = make(map[string]ent, len(items))
	for key, obj := range items {
		c.index[key] = c.encode(obj)
	}
}

func (c *Cache) dropHot(key string) {
	c.lmu.Lock()
	if el, ok := c.hot[key]; ok {
		c.dropHotLocked(el)
	}
	c.lmu.Unlock()
}

func (c *Cache) dropHotLocked(el *list.Element) {
	he := el.Value.(*hotEnt)
	delete(c.hot, he.key)
	c.lru.Remove(el)
	c.hotBytes -= he.size
}

// Stats returns counters for tests and debugging.
func (c *Cache) Stats() (encoded, passed, arenaUsed, hotBytes int) {
	if c.arena != nil {
		arenaUsed = c.arena.Used()
	}
	c.lmu.Lock()
	hotBytes = c.hotBytes
	c.lmu.Unlock()
	return c.encoded, c.passed, arenaUsed, hotBytes
}
