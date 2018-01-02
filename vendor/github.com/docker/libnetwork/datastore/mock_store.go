package datastore

import (
	"errors"

	"github.com/docker/libkv/store"
	"github.com/docker/libnetwork/types"
)

var (
	// ErrNotImplmented exported
	ErrNotImplmented = errors.New("Functionality not implemented")
)

// MockData exported
type MockData struct {
	Data  []byte
	Index uint64
}

// MockStore exported
type MockStore struct {
	db map[string]*MockData
}

// NewMockStore creates a Map backed Datastore that is useful for mocking
func NewMockStore() *MockStore {
	db := make(map[string]*MockData)
	return &MockStore{db}
}

// Get the value at "key", returns the last modified index
// to use in conjunction to CAS calls
func (s *MockStore) Get(key string) (*store.KVPair, error) {
	mData := s.db[key]
	if mData == nil {
		return nil, nil
	}
	return &store.KVPair{Value: mData.Data, LastIndex: mData.Index}, nil

}

// Put a value at "key"
func (s *MockStore) Put(key string, value []byte, options *store.WriteOptions) error {
	mData := s.db[key]
	if mData == nil {
		mData = &MockData{value, 0}
	}
	mData.Index = mData.Index + 1
	s.db[key] = mData
	return nil
}

// Delete a value at "key"
func (s *MockStore) Delete(key string) error {
	delete(s.db, key)
	return nil
}

// Exists checks that the key exists inside the store
func (s *MockStore) Exists(key string) (bool, error) {
	_, ok := s.db[key]
	return ok, nil
}

// List gets a range of values at "directory"
func (s *MockStore) List(prefix string) ([]*store.KVPair, error) {
	return nil, ErrNotImplmented
}

// DeleteTree deletes a range of values at "directory"
func (s *MockStore) DeleteTree(prefix string) error {
	delete(s.db, prefix)
	return nil
}

// Watch a single key for modifications
func (s *MockStore) Watch(key string, stopCh <-chan struct{}) (<-chan *store.KVPair, error) {
	return nil, ErrNotImplmented
}

// WatchTree triggers a watch on a range of values at "directory"
func (s *MockStore) WatchTree(prefix string, stopCh <-chan struct{}) (<-chan []*store.KVPair, error) {
	return nil, ErrNotImplmented
}

// NewLock exposed
func (s *MockStore) NewLock(key string, options *store.LockOptions) (store.Locker, error) {
	return nil, ErrNotImplmented
}

// AtomicPut put a value at "key" if the key has not been
// modified in the meantime, throws an error if this is the case
func (s *MockStore) AtomicPut(key string, newValue []byte, previous *store.KVPair, options *store.WriteOptions) (bool, *store.KVPair, error) {
	mData := s.db[key]

	if previous == nil {
		if mData != nil {
			return false, nil, types.BadRequestErrorf("atomic put failed because key exists")
		} // Else OK.
	} else {
		if mData == nil {
			return false, nil, types.BadRequestErrorf("atomic put failed because key exists")
		}
		if mData != nil && mData.Index != previous.LastIndex {
			return false, nil, types.BadRequestErrorf("atomic put failed due to mismatched Index")
		} // Else OK.
	}
	err := s.Put(key, newValue, nil)
	if err != nil {
		return false, nil, err
	}
	return true, &store.KVPair{Key: key, Value: newValue, LastIndex: s.db[key].Index}, nil
}

// AtomicDelete deletes a value at "key" if the key has not
// been modified in the meantime, throws an error if this is the case
func (s *MockStore) AtomicDelete(key string, previous *store.KVPair) (bool, error) {
	mData := s.db[key]
	if mData != nil && mData.Index != previous.LastIndex {
		return false, types.BadRequestErrorf("atomic delete failed due to mismatched Index")
	}
	return true, s.Delete(key)
}

// Close closes the client connection
func (s *MockStore) Close() {
	return
}
