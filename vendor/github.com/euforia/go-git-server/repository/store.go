package repository

import (
	"fmt"
	"sync"
)

// RepositoryStore is the repository storage interface that should be implemented.
type RepositoryStore interface {
	GetRepo(string) (*Repository, error)
	CreateRepo(*Repository) error
	UpdateRepo(*Repository) error
	RemoveRepo(string) error
}

// MemRepoStore is a memory based repo datastore
type MemRepoStore struct {
	mu sync.Mutex
	m  map[string]*Repository
}

// NewMemRepoStore instantiates a new repo store
func NewMemRepoStore() *MemRepoStore {
	return &MemRepoStore{m: map[string]*Repository{}}
}

// GetRepo with the given id
func (mrs *MemRepoStore) GetRepo(id string) (*Repository, error) {
	if v, ok := mrs.m[id]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("not found: %s", id)
}

// CreateRepo with the given repo data
func (mrs *MemRepoStore) CreateRepo(repo *Repository) error {
	mrs.mu.Lock()
	defer mrs.mu.Unlock()

	if _, ok := mrs.m[repo.ID]; ok {
		return fmt.Errorf("repo exists: %s", repo.ID)
	}
	mrs.m[repo.ID] = repo
	return nil
}

// UpdateRepo with the given data
func (mrs *MemRepoStore) UpdateRepo(repo *Repository) error {
	mrs.mu.Lock()
	defer mrs.mu.Unlock()

	if _, ok := mrs.m[repo.ID]; ok {
		mrs.m[repo.ID] = repo
		return nil
	}
	return fmt.Errorf("repo not found: %s", repo.ID)
}

// RemoveRepo with the given id
func (mrs *MemRepoStore) RemoveRepo(id string) error {
	mrs.mu.Lock()
	defer mrs.mu.Unlock()

	if _, ok := mrs.m[id]; ok {
		delete(mrs.m, id)
		return nil
	}
	return fmt.Errorf("repo not found: %s", id)
}
