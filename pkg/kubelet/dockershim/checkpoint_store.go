package dockershim

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// CheckpointStore provides the interface for checkpoint storage backend
type CheckpointStore interface {
	Add(key string, data []byte) error
	Get(key string) ([]byte, error)
	Delete(key string) error
	List() ([]string, error)
}

// FileStore is an implementation of CheckpointStore interface which stores checkpoint in file.
type FileStore struct {
	path string
}

func (fstore *FileStore) Add(key string, data []byte) error {
	if _, err := os.Stat(fstore.path); os.IsNotExist(err) {
		if err = os.MkdirAll(fstore.path, 755); err != nil {
			return err
		}
	}
	return ioutil.WriteFile(filepath.Join(fstore.path, key), data, 0644)
}

func (fstore *FileStore) Get(key string) ([]byte, error) {
	return ioutil.ReadFile(filepath.Join(fstore.path, key))
}

func (fstore *FileStore) Delete(key string) error {
	return os.Remove(filepath.Join(fstore.path, key))
}

func (fstroe *FileStore) List() ([]string, error) {
	keys := make([]string, 0)
	files, err := ioutil.ReadDir(fstroe.path)
	if err != nil {
		return nil, err
	}
	for _, f := range files {
		keys = append(keys, f.Name())
	}
	return keys, nil
}

// MemStore is an implementation of CheckpointStore interface which stores checkpoint in memory.
type MemStore struct {
	mem map[string][]byte
}

func NewMemStore() CheckpointStore {
	return &MemStore{mem: make(map[string][]byte)}
}

func (mstore *MemStore) Add(key string, data []byte) error {
	mstore.mem[key] = data
	return nil
}

func (mstore *MemStore) Get(key string) ([]byte, error) {
	data, ok := mstore.mem[key]
	if !ok {
		return nil, fmt.Errorf("Sandbox $q Checkpoint could not be found", key)
	}
	return data, nil
}

func (mstore *MemStore) Delete(key string) error {
	delete(mstore.mem, key)
	return nil
}

func (mstore *MemStore) List() ([]string, error) {
	keys := make([]string, 0)
	for key := range mstore.mem {
		keys = append(keys, key)
	}
	return keys, nil
}
