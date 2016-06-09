package cache

import "sync"

type SyncStore struct {
	Store
	cond sync.Cond
}

func NewSyncStore(s Store) *SyncStore {
	return &SyncStore{Store: s}
}

func (s *SyncStore) Add(obj interface{}) error {
	err := s.Store.Add(obj)
	s.cond.Signal()
	return err
}

func (s *SyncStore) Update(obj interface{}) error {
	err := s.Store.Update(obj)
	s.cond.Signal()
	return err
}

func (s *SyncStore) Delete(obj interface{}) error {
	err := s.Store.Delete(obj)
	s.cond.Signal()
	return err
}

func (s *SyncStore) Sync() {
	s.cond.Wait()
}
