package subscriber

import (
	"expvar"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"
	"sync"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
)

// Statistics for the Subscriber service.
const (
	statPointsWritten = "pointsWritten"
	statWriteFailures = "writeFailures"
)

// PointsWriter is an interface for writing points to a subscription destination.
// Only WritePoints() needs to be satisfied.
type PointsWriter interface {
	WritePoints(p *cluster.WritePointsRequest) error
}

// unique set that identifies a given subscription
type subEntry struct {
	db   string
	rp   string
	name string
}

// Service manages forking the incoming data from InfluxDB
// to defined third party destinations.
// Subscriptions are defined per database and retention policy.
type Service struct {
	subs      map[subEntry]PointsWriter
	MetaStore interface {
		Databases() ([]meta.DatabaseInfo, error)
		WaitForDataChanged() error
	}
	NewPointsWriter func(u url.URL) (PointsWriter, error)
	Logger          *log.Logger
	statMap         *expvar.Map
	points          chan *cluster.WritePointsRequest
	wg              sync.WaitGroup
	closed          bool
	mu              sync.Mutex
}

// NewService returns a subscriber service with given settings
func NewService(c Config) *Service {
	return &Service{
		subs:            make(map[subEntry]PointsWriter),
		NewPointsWriter: newPointsWriter,
		Logger:          log.New(os.Stderr, "[subscriber] ", log.LstdFlags),
		statMap:         influxdb.NewStatistics("subscriber", "subscriber", nil),
		points:          make(chan *cluster.WritePointsRequest),
		closed:          true,
	}
}

// Open starts the subscription service.
func (s *Service) Open() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.MetaStore == nil {
		panic("no meta store")
	}

	s.closed = false

	// Perform initial update
	s.Update()

	s.wg.Add(1)
	go s.writePoints()
	// Do not wait for this goroutine since it block until a meta change occurs.
	go s.waitForMetaUpdates()

	s.Logger.Println("opened service")
	return nil
}

// Close terminates the subscription service
func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	close(s.points)
	s.closed = true
	s.wg.Wait()
	s.Logger.Println("closed service")
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

func (s *Service) waitForMetaUpdates() {
	for {
		err := s.MetaStore.WaitForDataChanged()
		if err != nil {
			s.Logger.Printf("error while waiting for meta data changes, err: %v\n", err)
			return
		}
		//Check that we haven't been closed before performing update.
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			s.Logger.Println("service closed not updating")
			return
		}
		s.mu.Unlock()
		s.Update()
	}

}

// Update will start new and stop deleted subscriptions.
func (s *Service) Update() error {
	dbis, err := s.MetaStore.Databases()
	if err != nil {
		return err
	}
	allEntries := make(map[subEntry]bool, 0)
	// Add in new subscriptions
	for _, dbi := range dbis {
		for _, rpi := range dbi.RetentionPolicies {
			for _, si := range rpi.Subscriptions {
				se := subEntry{
					db:   dbi.Name,
					rp:   rpi.Name,
					name: si.Name,
				}
				allEntries[se] = true
				if _, ok := s.subs[se]; ok {
					continue
				}
				sub, err := s.createSubscription(se, si.Mode, si.Destinations)
				if err != nil {
					return err
				}
				s.subs[se] = sub
			}
		}
	}

	// Remove deleted subs
	for se := range s.subs {
		if !allEntries[se] {
			delete(s.subs, se)
			s.Logger.Println("deleted old subscription for", se.db, se.rp)
		}
	}

	return nil
}

func (s *Service) createSubscription(se subEntry, mode string, destinations []string) (PointsWriter, error) {
	var bm BalanceMode
	switch mode {
	case "ALL":
		bm = ALL
	case "ANY":
		bm = ANY
	default:
		return nil, fmt.Errorf("unknown balance mode %q", mode)
	}
	writers := make([]PointsWriter, len(destinations))
	statMaps := make([]*expvar.Map, len(writers))
	for i, dest := range destinations {
		u, err := url.Parse(dest)
		if err != nil {
			return nil, err
		}
		w, err := s.NewPointsWriter(*u)
		if err != nil {
			return nil, err
		}
		writers[i] = w
		tags := map[string]string{
			"database":         se.db,
			"retention_policy": se.rp,
			"name":             se.name,
			"mode":             mode,
			"destination":      dest,
		}
		key := strings.Join([]string{"subscriber", se.db, se.rp, se.name, dest}, ":")
		statMaps[i] = influxdb.NewStatistics(key, "subscriber", tags)
	}
	s.Logger.Println("created new subscription for", se.db, se.rp)
	return &balancewriter{
		bm:       bm,
		writers:  writers,
		statMaps: statMaps,
	}, nil
}

// Points returns a channel into which write point requests can be sent.
func (s *Service) Points() chan<- *cluster.WritePointsRequest {
	return s.points
}

// read points off chan and write them
func (s *Service) writePoints() {
	defer s.wg.Done()
	for p := range s.points {
		for se, sub := range s.subs {
			if p.Database == se.db && p.RetentionPolicy == se.rp {
				err := sub.WritePoints(p)
				if err != nil {
					s.Logger.Println(err)
					s.statMap.Add(statWriteFailures, 1)
				}
			}
		}
		s.statMap.Add(statPointsWritten, int64(len(p.Points)))
	}
}

// BalanceMode sets what balance mode to use on a subscription.
// valid options are currently ALL or ANY
type BalanceMode int

//ALL is a Balance mode option
const (
	ALL BalanceMode = iota
	ANY
)

// balances writes across PointsWriters according to BalanceMode
type balancewriter struct {
	bm       BalanceMode
	writers  []PointsWriter
	statMaps []*expvar.Map
	i        int
}

func (b *balancewriter) WritePoints(p *cluster.WritePointsRequest) error {
	var lastErr error
	for range b.writers {
		// round robin through destinations.
		i := b.i
		w := b.writers[i]
		b.i = (b.i + 1) % len(b.writers)

		// write points to destination.
		err := w.WritePoints(p)
		if err != nil {
			lastErr = err
			b.statMaps[i].Add(statWriteFailures, 1)
		} else {
			b.statMaps[i].Add(statPointsWritten, int64(len(p.Points)))
			if b.bm == ANY {
				break
			}
		}
	}
	return lastErr
}

// Creates a PointsWriter from the given URL
func newPointsWriter(u url.URL) (PointsWriter, error) {
	switch u.Scheme {
	case "udp":
		return NewUDP(u.Host), nil
	default:
		return nil, fmt.Errorf("unknown destination scheme %s", u.Scheme)
	}
}
