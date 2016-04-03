package retention

import (
	"log"
	"os"
	"sync"
	"time"

	"github.com/influxdb/influxdb/meta"
)

// Service represents the retention policy enforcement service.
type Service struct {
	MetaStore interface {
		IsLeader() bool
		VisitRetentionPolicies(f func(d meta.DatabaseInfo, r meta.RetentionPolicyInfo))
		DeleteShardGroup(database, policy string, id uint64) error
	}
	TSDBStore interface {
		ShardIDs() []uint64
		DeleteShard(shardID uint64) error
	}

	enabled       bool
	checkInterval time.Duration
	wg            sync.WaitGroup
	done          chan struct{}

	logger *log.Logger
}

// NewService returns a configure retention policy enforcement service.
func NewService(c Config) *Service {
	return &Service{
		checkInterval: time.Duration(c.CheckInterval),
		done:          make(chan struct{}),
		logger:        log.New(os.Stderr, "[retention] ", log.LstdFlags),
	}
}

// Open starts retention policy enforcement.
func (s *Service) Open() error {
	s.wg.Add(2)
	go s.deleteShardGroups()
	go s.deleteShards()
	return nil
}

// Close stops retention policy enforcement.
func (s *Service) Close() error {
	close(s.done)
	s.wg.Wait()
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.logger = l
}

func (s *Service) deleteShardGroups() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.checkInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			s.logger.Println("retention policy enforcement terminating")
			return

		case <-ticker.C:
			// Only run this on the leader, but always allow the loop to check
			// as the leader can change.
			if !s.MetaStore.IsLeader() {
				continue
			}
			s.logger.Println("retention policy enforcement check commencing")

			s.MetaStore.VisitRetentionPolicies(func(d meta.DatabaseInfo, r meta.RetentionPolicyInfo) {
				for _, g := range r.ExpiredShardGroups(time.Now().UTC()) {
					if err := s.MetaStore.DeleteShardGroup(d.Name, r.Name, g.ID); err != nil {
						s.logger.Printf("failed to delete shard group %d from database %s, retention policy %s: %s",
							g.ID, d.Name, r.Name, err.Error())
					} else {
						s.logger.Printf("deleted shard group %d from database %s, retention policy %s",
							g.ID, d.Name, r.Name)
					}
				}
			})
		}
	}
}

func (s *Service) deleteShards() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.checkInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			s.logger.Println("retention policy enforcement terminating")
			return

		case <-ticker.C:
			s.logger.Println("retention policy shard deletion check commencing")

			deletedShardIDs := make(map[uint64]struct{}, 0)
			s.MetaStore.VisitRetentionPolicies(func(d meta.DatabaseInfo, r meta.RetentionPolicyInfo) {
				for _, g := range r.DeletedShardGroups() {
					for _, sh := range g.Shards {
						deletedShardIDs[sh.ID] = struct{}{}
					}
				}
			})

			for _, id := range s.TSDBStore.ShardIDs() {
				if _, ok := deletedShardIDs[id]; ok {
					if err := s.TSDBStore.DeleteShard(id); err != nil {
						s.logger.Printf("failed to delete shard ID %d: %s", id, err.Error())
						continue
					}
					s.logger.Printf("shard ID %d deleted", id)
				}
			}
		}
	}
}
