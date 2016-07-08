package utils

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	log "github.com/Sirupsen/logrus"

	"github.com/emccode/libstorage/api/types"
)

type keyValueStore struct {
	sync.RWMutex
	store  map[string]interface{}
	ttl    time.Duration
	timers map[string]*ttlTimer
	hl     bool
}

func (s *keyValueStore) String() string {
	s.RLock()
	defer s.RUnlock()
	return fmt.Sprintf("%v", s.store)
}

type ttlTimer struct {
	sync.RWMutex
	expires *time.Time
}

func (t *ttlTimer) touch(duration time.Duration) {
	t.Lock()
	defer t.Unlock()
	expiration := time.Now().Add(duration)
	t.expires = &expiration
}

func (t *ttlTimer) expired() bool {
	t.RLock()
	defer t.RUnlock()
	if t.expires == nil {
		return true
	}
	return t.expires.Before(time.Now())
}

// NewStore initializes a new instance of the Store type.
func NewStore() types.Store {
	return newKeyValueStore(map[string]interface{}{}, -1, false)
}

// NewTTLStore initializes a new instance of the Store type, but has a TTL
// that expires contents after a specific duration. The parameter hardLimit
// can be set to true to change the TTL from a sliding expiration to a hard
// limit.
func NewTTLStore(duration time.Duration, hardLimit bool) types.Store {
	return newKeyValueStore(map[string]interface{}{}, duration, hardLimit)
}

// NewStoreWithData initializes a new instance of the Store type.
func NewStoreWithData(data map[string]interface{}) types.Store {
	return newKeyValueStore(data, -1, false)
}

// NewStoreWithVars initializes a new instance of the Store type.
func NewStoreWithVars(vars map[string]string) types.Store {
	m := map[string]interface{}{}
	for k, v := range vars {
		m[k] = v
	}
	return newKeyValueStore(m, -1, false)
}

func newKeyValueStore(
	m map[string]interface{},
	ttl time.Duration,
	hardLimit bool) types.Store {

	cm := map[string]interface{}{}
	for k, v := range m {
		cm[strings.ToLower(k)] = v
	}
	store := &keyValueStore{store: cm}
	if ttl > -1 {
		store.hl = hardLimit
		store.ttl = ttl
		store.timers = map[string]*ttlTimer{}
		for k := range cm {
			store.timers[k].touch(store.ttl)
		}
		store.startCleanupTimer()
	}
	return store
}

func (s *keyValueStore) cleanup() {
	s.Lock()
	defer s.Unlock()
	for k, v := range s.timers {
		if v.expired() {
			delete(s.store, k)
			delete(s.timers, k)
			log.WithField("key", k).Debug("expiring cached key")
		}
	}
}

func (s *keyValueStore) startCleanupTimer() {
	duration := s.ttl
	if duration < time.Second {
		duration = time.Second
	}
	ticker := time.Tick(duration)
	go (func() {
		for {
			select {
			case <-ticker:
				s.cleanup()
			}
		}
	})()
}

func (s *keyValueStore) Delete(k string) interface{} {
	s.Lock()
	defer s.Unlock()
	k = strings.ToLower(k)
	if v, ok := s.store[k]; ok {
		delete(s.store, k)
		return v
	}
	return nil
}

func (s *keyValueStore) Map() map[string]interface{} {
	s.RLock()
	defer s.RUnlock()
	m := map[string]interface{}{}
	for _, k := range s.Keys() {
		m[k] = s.Get(k)
	}
	return m
}

func (s *keyValueStore) IsSet(k string) bool {
	s.RLock()
	defer s.RUnlock()
	k = strings.ToLower(k)
	_, vok := s.store[k]
	if s.timers == nil || !vok {
		return vok
	}
	ttlt, tok := s.timers[k]
	if !tok || ttlt.expired() {
		return false
	}
	if !s.hl {
		ttlt.touch(s.ttl)
	}
	return true
}

func (s *keyValueStore) Get(k string) interface{} {
	s.RLock()
	defer s.RUnlock()
	k = strings.ToLower(k)
	v, vok := s.store[k]
	if s.timers == nil || !vok {
		return v
	}
	ttlt, tok := s.timers[k]
	if !tok || ttlt.expired() {
		return nil
	}
	if !s.hl {
		ttlt.touch(s.ttl)
	}
	return v
}

func (s *keyValueStore) GetStore(k string) types.Store {
	v := s.Get(k)
	switch tv := v.(type) {
	case types.Store:
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) GetString(k string) string {
	v := s.Get(k)
	switch tv := v.(type) {
	case string:
		return tv
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", tv)
	}
}

func (s *keyValueStore) GetStringPtr(k string) *string {
	v := s.Get(k)
	switch tv := v.(type) {
	case *string:
		return tv
	case string:
		return &tv
	case nil:
		return nil
	default:
		str := getStrFromPossiblePtr(v)
		return &str
	}
}

func (s *keyValueStore) GetBool(k string) bool {
	v := s.Get(k)
	switch tv := v.(type) {
	case bool:
		return tv
	case nil:
		return false
	default:
		b, _ := strconv.ParseBool(s.GetString(k))
		return b
	}
}

func (s *keyValueStore) GetBoolPtr(k string) *bool {
	v := s.Get(k)
	switch tv := v.(type) {
	case *bool:
		return tv
	case bool:
		return &tv
	case nil:
		return nil
	default:
		str := getStrFromPossiblePtr(v)
		b, _ := strconv.ParseBool(str)
		return &b
	}
}

func (s *keyValueStore) GetInt(k string) int {
	v := s.Get(k)
	switch tv := v.(type) {
	case int:
		return tv
	case nil:
		return 0
	default:
		if iv, err := strconv.ParseInt(s.GetString(k), 10, 64); err == nil {
			return int(iv)
		}
		return 0
	}
}

func (s *keyValueStore) GetIntPtr(k string) *int {
	v := s.Get(k)
	switch tv := v.(type) {
	case *int:
		return tv
	case int:
		return &tv
	case nil:
		return nil
	default:
		str := getStrFromPossiblePtr(v)
		var iivp *int
		if iv, err := strconv.ParseInt(str, 10, 64); err == nil {
			iiv := int(iv)
			iivp = &iiv
		}
		return iivp
	}
}

func (s *keyValueStore) GetInt64(k string) int64 {
	v := s.Get(k)
	switch tv := v.(type) {
	case int64:
		return tv
	case nil:
		return 0
	default:
		if iv, err := strconv.ParseInt(s.GetString(k), 10, 64); err == nil {
			return iv
		}
		return 0
	}
}

func (s *keyValueStore) GetInt64Ptr(k string) *int64 {
	v := s.Get(k)
	switch tv := v.(type) {
	case *int64:
		return tv
	case int64:
		return &tv
	case nil:
		return nil
	default:
		str := getStrFromPossiblePtr(v)
		var ivp *int64
		if iv, err := strconv.ParseInt(str, 10, 64); err == nil {
			ivp = &iv
		}
		return ivp
	}
}

func (s *keyValueStore) GetStringSlice(k string) []string {
	v := s.Get(k)
	switch tv := v.(type) {
	case []string:
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) GetIntSlice(k string) []int {
	v := s.Get(k)
	switch tv := v.(type) {
	case []int:
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) GetBoolSlice(k string) []bool {
	v := s.Get(k)
	switch tv := v.(type) {
	case []bool:
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) GetMap(k string) map[string]interface{} {
	v := s.Get(k)
	switch tv := v.(type) {
	case map[string]interface{}:
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) GetInstanceID(k string) *types.InstanceID {
	v := s.Get(k)
	switch tv := v.(type) {
	case (*types.InstanceID):
		return tv
	default:
		return nil
	}
}

func (s *keyValueStore) Set(k string, v interface{}) {
	s.Lock()
	defer s.Unlock()
	if s.timers != nil {
		ttlt := &ttlTimer{}
		ttlt.touch(s.ttl)
		s.timers[strings.ToLower(k)] = ttlt
	}
	s.store[strings.ToLower(k)] = v
}

func (s *keyValueStore) Keys() []string {
	s.RLock()
	defer s.RUnlock()
	keys := []string{}
	for k := range s.store {
		keys = append(keys, k)
	}
	return keys
}

func getStrFromPossiblePtr(i interface{}) string {
	rv := reflect.ValueOf(i)
	if rv.Kind() == reflect.Ptr {
		return fmt.Sprintf("%v", rv.Elem())
	}
	return fmt.Sprintf("%v", i)
}
