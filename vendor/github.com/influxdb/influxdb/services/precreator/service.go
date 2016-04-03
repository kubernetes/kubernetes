package precreator

import (
	"log"
	"os"
	"sync"
	"time"
)

type Service struct {
	checkInterval time.Duration
	advancePeriod time.Duration

	Logger *log.Logger

	done chan struct{}
	wg   sync.WaitGroup

	MetaStore interface {
		IsLeader() bool
		PrecreateShardGroups(cutoff time.Time) error
	}
}

// NewService returns an instance of the precreation service.
func NewService(c Config) (*Service, error) {
	s := Service{
		checkInterval: time.Duration(c.CheckInterval),
		advancePeriod: time.Duration(c.AdvancePeriod),
		Logger:        log.New(os.Stderr, "[shard-precreation] ", log.LstdFlags),
	}

	return &s, nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// Open starts the precreation service.
func (s *Service) Open() error {
	if s.done != nil {
		return nil
	}

	s.done = make(chan struct{})

	s.wg.Add(1)
	go s.runPrecreation()
	return nil
}

// Close stops the precreation service.
func (s *Service) Close() error {
	if s.done == nil {
		return nil
	}

	close(s.done)
	s.wg.Wait()
	s.done = nil

	return nil
}

// runPrecreation continually checks if resources need precreation.
func (s *Service) runPrecreation() {
	defer s.wg.Done()

	for {
		select {
		case <-time.After(s.checkInterval):
			// Only run this on the leader, but always allow the loop to check
			// as the leader can change.
			if !s.MetaStore.IsLeader() {
				continue
			}

			if err := s.precreate(time.Now().UTC()); err != nil {
				s.Logger.Printf("failed to precreate shards: %s", err.Error())
			}
		case <-s.done:
			s.Logger.Println("precreation service terminating")
			return
		}
	}
}

// precreate performs actual resource precreation.
func (s *Service) precreate(t time.Time) error {
	cutoff := t.Add(s.advancePeriod).UTC()
	if err := s.MetaStore.PrecreateShardGroups(cutoff); err != nil {
		return err
	}
	return nil
}
