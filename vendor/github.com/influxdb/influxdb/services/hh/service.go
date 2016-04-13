package hh

import (
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"

	"github.com/influxdb/influxdb/tsdb"
)

var ErrHintedHandoffDisabled = fmt.Errorf("hinted handoff disabled")

type Service struct {
	mu      sync.RWMutex
	wg      sync.WaitGroup
	closing chan struct{}

	Logger *log.Logger
	cfg    Config

	ShardWriter shardWriter

	HintedHandoff interface {
		WriteShard(shardID, ownerID uint64, points []tsdb.Point) error
		Process() error
		PurgeOlderThan(when time.Duration) error
	}
}

type shardWriter interface {
	WriteShard(shardID, ownerID uint64, points []tsdb.Point) error
}

// NewService returns a new instance of Service.
func NewService(c Config, w shardWriter) *Service {
	s := &Service{
		cfg:    c,
		Logger: log.New(os.Stderr, "[handoff] ", log.LstdFlags),
	}
	processor, err := NewProcessor(c.Dir, w, ProcessorOptions{
		MaxSize:        c.MaxSize,
		RetryRateLimit: c.RetryRateLimit,
	})
	if err != nil {
		s.Logger.Fatalf("Failed to start hinted handoff processor: %v", err)
	}
	processor.Logger = s.Logger
	s.HintedHandoff = processor
	return s
}

func (s *Service) Open() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.closing = make(chan struct{})

	s.wg.Add(2)
	go s.retryWrites()
	go s.expireWrites()

	return nil
}

func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closing != nil {
		close(s.closing)
	}
	s.wg.Wait()
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// WriteShard queues the points write for shardID to node ownerID to handoff queue
func (s *Service) WriteShard(shardID, ownerID uint64, points []tsdb.Point) error {
	if !s.cfg.Enabled {
		return ErrHintedHandoffDisabled
	}

	return s.HintedHandoff.WriteShard(shardID, ownerID, points)
}

func (s *Service) retryWrites() {
	defer s.wg.Done()
	ticker := time.NewTicker(time.Duration(s.cfg.RetryInterval))
	defer ticker.Stop()
	for {
		select {
		case <-s.closing:
			return
		case <-ticker.C:
			if err := s.HintedHandoff.Process(); err != nil && err != io.EOF {
				s.Logger.Printf("retried write failed: %v", err)
			}
		}
	}
}

// expireWrites will cause the handoff queues to remove writes that are older
// than the configured threshold
func (s *Service) expireWrites() {
	defer s.wg.Done()
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	for {
		select {
		case <-s.closing:
			return
		case <-ticker.C:
			if err := s.HintedHandoff.PurgeOlderThan(time.Duration(s.cfg.MaxAge)); err != nil {
				s.Logger.Printf("purge write failed: %v", err)
			}
		}
	}
}

// purgeWrites will cause the handoff queues to remove writes that are no longer
// valid.  e.g. queued writes for a node that has been removed
func (s *Service) purgeWrites() {
	panic("not implemented")
}
