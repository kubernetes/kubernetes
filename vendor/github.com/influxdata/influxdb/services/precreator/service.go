package precreator // import "github.com/influxdata/influxdb/services/precreator"

import (
	"io"
	"log"
	"os"
	"sync"
	"time"
)

// Service manages the shard precreation service.
type Service struct {
	checkInterval time.Duration
	advancePeriod time.Duration

	Logger *log.Logger

	done chan struct{}
	wg   sync.WaitGroup

	MetaClient interface {
		PrecreateShardGroups(now, cutoff time.Time) error
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

// SetLogOutput sets the writer to which all logs are written. It must not be
// called after Open is called.
func (s *Service) SetLogOutput(w io.Writer) {
	s.Logger = log.New(w, "[shard-precreation] ", log.LstdFlags)
}

// Open starts the precreation service.
func (s *Service) Open() error {
	if s.done != nil {
		return nil
	}

	s.Logger.Printf("Starting precreation service with check interval of %s, advance period of %s",
		s.checkInterval, s.advancePeriod)

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
			if err := s.precreate(time.Now().UTC()); err != nil {
				s.Logger.Printf("failed to precreate shards: %s", err.Error())
			}
		case <-s.done:
			s.Logger.Println("Precreation service terminating")
			return
		}
	}
}

// precreate performs actual resource precreation.
func (s *Service) precreate(now time.Time) error {
	cutoff := now.Add(s.advancePeriod).UTC()
	if err := s.MetaClient.PrecreateShardGroups(now, cutoff); err != nil {
		return err
	}
	return nil
}
