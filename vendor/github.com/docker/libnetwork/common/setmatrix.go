package common

import (
	"sync"

	mapset "github.com/deckarep/golang-set"
)

// SetMatrix is a map of Sets
type SetMatrix interface {
	// Get returns the members of the set for a specific key as a slice.
	Get(key string) ([]interface{}, bool)
	// Contains is used to verify if an element is in a set for a specific key
	// returns true if the element is in the set
	// returns true if there is a set for the key
	Contains(key string, value interface{}) (bool, bool)
	// Insert inserts the value in the set of a key
	// returns true if the value is inserted (was not already in the set), false otherwise
	// returns also the length of the set for the key
	Insert(key string, value interface{}) (bool, int)
	// Remove removes the value in the set for a specific key
	// returns true if the value is deleted, false otherwise
	// returns also the length of the set for the key
	Remove(key string, value interface{}) (bool, int)
	// Cardinality returns the number of elements in the set for a key
	// returns false if the set is not present
	Cardinality(key string) (int, bool)
	// String returns the string version of the set, empty otherwise
	// returns false if the set is not present
	String(key string) (string, bool)
	// Returns all the keys in the map
	Keys() []string
}

type setMatrix struct {
	matrix map[string]mapset.Set

	sync.Mutex
}

// NewSetMatrix creates a new set matrix object
func NewSetMatrix() SetMatrix {
	s := &setMatrix{}
	s.init()
	return s
}

func (s *setMatrix) init() {
	s.matrix = make(map[string]mapset.Set)
}

func (s *setMatrix) Get(key string) ([]interface{}, bool) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		return nil, ok
	}
	return set.ToSlice(), ok
}

func (s *setMatrix) Contains(key string, value interface{}) (bool, bool) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		return false, ok
	}
	return set.Contains(value), ok
}

func (s *setMatrix) Insert(key string, value interface{}) (bool, int) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		s.matrix[key] = mapset.NewSet()
		s.matrix[key].Add(value)
		return true, 1
	}

	return set.Add(value), set.Cardinality()
}

func (s *setMatrix) Remove(key string, value interface{}) (bool, int) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		return false, 0
	}

	var removed bool
	if set.Contains(value) {
		set.Remove(value)
		removed = true
		// If the set is empty remove it from the matrix
		if set.Cardinality() == 0 {
			delete(s.matrix, key)
		}
	}

	return removed, set.Cardinality()
}

func (s *setMatrix) Cardinality(key string) (int, bool) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		return 0, ok
	}

	return set.Cardinality(), ok
}

func (s *setMatrix) String(key string) (string, bool) {
	s.Lock()
	defer s.Unlock()
	set, ok := s.matrix[key]
	if !ok {
		return "", ok
	}
	return set.String(), ok
}

func (s *setMatrix) Keys() []string {
	s.Lock()
	defer s.Unlock()
	keys := make([]string, 0, len(s.matrix))
	for k := range s.matrix {
		keys = append(keys, k)
	}
	return keys
}
