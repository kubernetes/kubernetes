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
	if s.cond.L != nil {
		s.cond.Signal()
	}
	return err
}

func (s *SyncStore) Update(obj interface{}) error {
	err := s.Store.Update(obj)
	if s.cond.L != nil {
		s.cond.Signal()
	}
	return err
}

func (s *SyncStore) Delete(obj interface{}) error {
	err := s.Store.Delete(obj)
	if s.cond.L != nil {
		s.cond.Signal()
	}
	return err
}

func (s *SyncStore) Sync() {
	if s.cond.L != nil {
		s.cond.Wait()
	}
}
