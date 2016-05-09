package tsdb

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/influxdb/influxdb/influxql"
)

func NewStore(path string) *Store {
	return &Store{
		path:                   path,
		MaxWALSize:             DefaultMaxWALSize,
		WALFlushInterval:       DefaultWALFlushInterval,
		WALPartitionFlushDelay: DefaultWALPartitionFlushDelay,
		Logger:                 log.New(os.Stderr, "[store] ", log.LstdFlags),
	}
}

var (
	ErrShardNotFound = fmt.Errorf("shard not found")
)

type Store struct {
	mu   sync.RWMutex
	path string

	databaseIndexes map[string]*DatabaseIndex
	shards          map[uint64]*Shard

	MaxWALSize             int
	WALFlushInterval       time.Duration
	WALPartitionFlushDelay time.Duration

	Logger *log.Logger
}

// Path returns the store's root path.
func (s *Store) Path() string { return s.path }

func (s *Store) CreateShard(database, retentionPolicy string, shardID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// shard already exists
	if _, ok := s.shards[shardID]; ok {
		return nil
	}

	// created the db and retention policy dirs if they don't exist
	if err := os.MkdirAll(filepath.Join(s.path, database, retentionPolicy), 0700); err != nil {
		return err
	}

	// create the database index if it does not exist
	db, ok := s.databaseIndexes[database]
	if !ok {
		db = NewDatabaseIndex()
		s.databaseIndexes[database] = db
	}

	shardPath := filepath.Join(s.path, database, retentionPolicy, strconv.FormatUint(shardID, 10))
	shard := s.newShard(db, shardPath)
	if err := shard.Open(); err != nil {
		return err
	}

	s.shards[shardID] = shard

	return nil
}

// DeleteShard removes a shard from disk.
func (s *Store) DeleteShard(shardID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// ensure shard exists
	sh, ok := s.shards[shardID]
	if !ok {
		return nil
	}

	if err := sh.Close(); err != nil {
		return err
	}

	if err := os.Remove(sh.path); err != nil {
		return err
	}

	delete(s.shards, shardID)

	return nil
}

// newShard returns a shard and copies configuration settings from the store.
func (s *Store) newShard(index *DatabaseIndex, path string) *Shard {
	sh := NewShard(index, path)
	sh.MaxWALSize = s.MaxWALSize
	sh.WALFlushInterval = s.WALFlushInterval
	sh.WALPartitionFlushDelay = s.WALPartitionFlushDelay
	return sh
}

// DeleteDatabase will close all shards associated with a database and remove the directory and files from disk.
func (s *Store) DeleteDatabase(name string, shardIDs []uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range shardIDs {
		shard := s.shards[id]
		if shard != nil {
			shard.Close()
		}
	}
	if err := os.RemoveAll(filepath.Join(s.path, name)); err != nil {
		return err
	}
	delete(s.databaseIndexes, name)
	return nil
}

func (s *Store) Shard(shardID uint64) *Shard {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.shards[shardID]
}

// ShardIDs returns a slice of all ShardIDs under management.
func (s *Store) ShardIDs() []uint64 {
	ids := make([]uint64, 0, len(s.shards))
	for i, _ := range s.shards {
		ids = append(ids, i)
	}
	return ids
}

func (s *Store) ValidateAggregateFieldsInStatement(shardID uint64, measurementName string, stmt *influxql.SelectStatement) error {
	s.mu.RLock()
	shard := s.shards[shardID]
	s.mu.RUnlock()
	if shard == nil {
		return ErrShardNotFound
	}
	return shard.ValidateAggregateFieldsInStatement(measurementName, stmt)
}

func (s *Store) DatabaseIndex(name string) *DatabaseIndex {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.databaseIndexes[name]
}

func (s *Store) Measurement(database, name string) *Measurement {
	s.mu.RLock()
	db := s.databaseIndexes[database]
	s.mu.RUnlock()
	if db == nil {
		return nil
	}
	return db.Measurement(name)
}

// deleteSeries loops through the local shards and deletes the series data and metadata for the passed in series keys
func (s *Store) deleteSeries(keys []string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, sh := range s.shards {
		if err := sh.deleteSeries(keys); err != nil {
			return err
		}
	}
	return nil
}

// deleteMeasurement loops through the local shards and removes the measurement field encodings from each shard
func (s *Store) deleteMeasurement(name string, seriesKeys []string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, sh := range s.shards {
		if err := sh.deleteMeasurement(name, seriesKeys); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) loadIndexes() error {
	dbs, err := ioutil.ReadDir(s.path)
	if err != nil {
		return err
	}
	for _, db := range dbs {
		if !db.IsDir() {
			s.Logger.Printf("Skipping database dir: %s. Not a directory", db.Name())
			continue
		}
		s.databaseIndexes[db.Name()] = NewDatabaseIndex()
	}
	return nil
}

func (s *Store) loadShards() error {
	// loop through the current database indexes
	for db := range s.databaseIndexes {
		rps, err := ioutil.ReadDir(filepath.Join(s.path, db))
		if err != nil {
			return err
		}

		for _, rp := range rps {
			// retention policies should be directories.  Skip anything that is not a dir.
			if !rp.IsDir() {
				s.Logger.Printf("Skipping retention policy dir: %s. Not a directory", rp.Name())
				continue
			}

			shards, err := ioutil.ReadDir(filepath.Join(s.path, db, rp.Name()))
			if err != nil {
				return err
			}
			for _, sh := range shards {
				path := filepath.Join(s.path, db, rp.Name(), sh.Name())

				// Shard file names are numeric shardIDs
				shardID, err := strconv.ParseUint(sh.Name(), 10, 64)
				if err != nil {
					s.Logger.Printf("Skipping shard: %s. Not a valid path", rp.Name())
					continue
				}

				shard := s.newShard(s.databaseIndexes[db], path)
				shard.Open()
				s.shards[shardID] = shard
			}
		}
	}
	return nil

}

func (s *Store) Open() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.shards = map[uint64]*Shard{}
	s.databaseIndexes = map[string]*DatabaseIndex{}

	// Create directory.
	if err := os.MkdirAll(s.path, 0777); err != nil {
		return err
	}

	// TODO: Start AE for Node
	if err := s.loadIndexes(); err != nil {
		return err
	}

	if err := s.loadShards(); err != nil {
		return err
	}

	return nil
}

func (s *Store) WriteToShard(shardID uint64, points []Point) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	sh, ok := s.shards[shardID]
	if !ok {
		return ErrShardNotFound
	}

	return sh.WritePoints(points)
}

// Flush forces all shards to write their WAL data to the index.
func (s *Store) Flush() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for shardID, sh := range s.shards {
		if err := sh.Flush(s.WALPartitionFlushDelay); err != nil {
			return fmt.Errorf("flush: shard=%d, err=%s", shardID, err)
		}
	}
	return nil
}

func (s *Store) CreateMapper(shardID uint64, query string, chunkSize int) (Mapper, error) {
	q, err := influxql.NewParser(strings.NewReader(query)).ParseStatement()
	if err != nil {
		return nil, err
	}
	stmt, ok := q.(*influxql.SelectStatement)
	if !ok {
		return nil, fmt.Errorf("query is not a SELECT statement: %s", err.Error())
	}

	shard := s.Shard(shardID)
	if shard == nil {
		// This can happen if the shard has been assigned, but hasn't actually been created yet.
		return nil, nil
	}

	if (stmt.IsRawQuery && !stmt.HasDistinct()) || stmt.IsSimpleDerivative() {
		return NewRawMapper(shard, stmt, chunkSize), nil
	}
	return NewAggMapper(shard, stmt), nil
}

func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, sh := range s.shards {
		if err := sh.Close(); err != nil {
			return err
		}
	}
	s.shards = nil
	s.databaseIndexes = nil

	return nil
}

// IsRetryable returns true if this error is temporary and could be retried
func IsRetryable(err error) bool {
	if err == nil {
		return true
	}

	if strings.Contains(err.Error(), "field type conflict") {
		return false
	}
	return true
}
