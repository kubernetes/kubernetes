package collectd

import (
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/kimor79/gollectd"
)

const leaderWaitTimeout = 30 * time.Second

// pointsWriter is an internal interface to make testing easier.
type pointsWriter interface {
	WritePoints(p *cluster.WritePointsRequest) error
}

// metaStore is an internal interface to make testing easier.
type metaStore interface {
	WaitForLeader(d time.Duration) error
	CreateDatabaseIfNotExists(name string) (*meta.DatabaseInfo, error)
}

// Service represents a UDP server which receives metrics in collectd's binary
// protocol and stores them in InfluxDB.
type Service struct {
	Config       *Config
	MetaStore    metaStore
	PointsWriter pointsWriter
	Logger       *log.Logger

	wg      sync.WaitGroup
	err     chan error
	stop    chan struct{}
	ln      *net.UDPConn
	batcher *tsdb.PointBatcher
	typesdb gollectd.Types
	addr    net.Addr
}

// NewService returns a new instance of the collectd service.
func NewService(c Config) *Service {
	s := &Service{
		Config: &c,
		Logger: log.New(os.Stderr, "[collectd] ", log.LstdFlags),
		err:    make(chan error),
	}

	return s
}

// Open starts the service.
func (s *Service) Open() error {
	if s.Config.BindAddress == "" {
		return fmt.Errorf("bind address is blank")
	} else if s.Config.Database == "" {
		return fmt.Errorf("database name is blank")
	} else if s.PointsWriter == nil {
		return fmt.Errorf("PointsWriter is nil")
	}

	if err := s.MetaStore.WaitForLeader(leaderWaitTimeout); err != nil {
		s.Logger.Printf("failed to detect a cluster leader: %s", err.Error())
		return err
	}

	if _, err := s.MetaStore.CreateDatabaseIfNotExists(s.Config.Database); err != nil {
		s.Logger.Printf("failed to ensure target database %s exists: %s", s.Config.Database, err.Error())
		return err
	}

	if s.typesdb == nil {
		// Open collectd types.
		typesdb, err := gollectd.TypesDBFile(s.Config.TypesDB)
		if err != nil {
			return fmt.Errorf("Open(): %s", err)
		}
		s.typesdb = typesdb
	}

	// Resolve our address.
	addr, err := net.ResolveUDPAddr("udp", s.Config.BindAddress)
	if err != nil {
		return fmt.Errorf("unable to resolve UDP address: %s", err)
	}
	s.addr = addr

	// Start listening
	ln, err := net.ListenUDP("udp", addr)
	if err != nil {
		return fmt.Errorf("unable to listen on UDP: %s", err)
	}
	s.ln = ln

	// Start the points batcher.
	s.batcher = tsdb.NewPointBatcher(s.Config.BatchSize, time.Duration(s.Config.BatchDuration))
	s.batcher.Start()

	// Create channel and wait group for signalling goroutines to stop.
	s.stop = make(chan struct{})
	s.wg.Add(2)

	// Start goroutines that process collectd packets.
	go s.serve()
	go s.writePoints()

	s.Logger.Println("collectd UDP started")

	return nil
}

// Close stops the service.
func (s *Service) Close() error {
	// Close the connection, and wait for the goroutine to exit.
	if s.stop != nil {
		close(s.stop)
	}
	if s.ln != nil {
		s.ln.Close()
	}
	if s.batcher != nil {
		s.batcher.Stop()
	}
	s.wg.Wait()

	// Release all remaining resources.
	s.stop = nil
	s.ln = nil
	s.batcher = nil
	s.Logger.Println("collectd UDP closed")
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// SetTypes sets collectd types db.
func (s *Service) SetTypes(types string) (err error) {
	s.typesdb, err = gollectd.TypesDB([]byte(types))
	return
}

// Err returns a channel for fatal errors that occur on go routines.
func (s *Service) Err() chan error { return s.err }

// Addr returns the listener's address. Returns nil if listener is closed.
func (s *Service) Addr() net.Addr {
	return s.ln.LocalAddr()
}

func (s *Service) serve() {
	defer s.wg.Done()

	// From https://collectd.org/wiki/index.php/Binary_protocol
	//   1024 bytes (payload only, not including UDP / IP headers)
	//   In versions 4.0 through 4.7, the receive buffer has a fixed size
	//   of 1024 bytes. When longer packets are received, the trailing data
	//   is simply ignored. Since version 4.8, the buffer size can be
	//   configured. Version 5.0 will increase the default buffer size to
	//   1452 bytes (the maximum payload size when using UDP/IPv6 over
	//   Ethernet).
	buffer := make([]byte, 1452)

	for {
		select {
		case <-s.stop:
			// We closed the connection, time to go.
			return
		default:
			// Keep processing.
		}

		n, _, err := s.ln.ReadFromUDP(buffer)
		if err != nil {
			s.Logger.Printf("collectd ReadFromUDP error: %s", err)
			continue
		}
		if n > 0 {
			s.handleMessage(buffer[:n])
		}
	}
}

func (s *Service) handleMessage(buffer []byte) {
	packets, err := gollectd.Packets(buffer, s.typesdb)
	if err != nil {
		s.Logger.Printf("Collectd parse error: %s", err)
		return
	}
	for _, packet := range *packets {
		points := Unmarshal(&packet)
		for _, p := range points {
			s.batcher.In() <- p
		}
	}
}

func (s *Service) writePoints() {
	defer s.wg.Done()

	for {
		select {
		case <-s.stop:
			return
		case batch := <-s.batcher.Out():
			req := &cluster.WritePointsRequest{
				Database:         s.Config.Database,
				RetentionPolicy:  s.Config.RetentionPolicy,
				ConsistencyLevel: cluster.ConsistencyLevelAny,
				Points:           batch,
			}
			if err := s.PointsWriter.WritePoints(req); err != nil {
				s.Logger.Printf("failed to write batch: %s", err)
				continue
			}
		}
	}
}

// Unmarshal translates a collectd packet into InfluxDB data points.
func Unmarshal(packet *gollectd.Packet) []tsdb.Point {
	// Prefer high resolution timestamp.
	var timestamp time.Time
	if packet.TimeHR > 0 {
		// TimeHR is "near" nanosecond measurement, but not exactly nanasecond time
		// Since we store time in microseconds, we round here (mostly so tests will work easier)
		sec := packet.TimeHR >> 30
		// Shifting, masking, and dividing by 1 billion to get nanoseconds.
		nsec := ((packet.TimeHR & 0x3FFFFFFF) << 30) / 1000 / 1000 / 1000
		timestamp = time.Unix(int64(sec), int64(nsec)).UTC().Round(time.Microsecond)
	} else {
		// If we don't have high resolution time, fall back to basic unix time
		timestamp = time.Unix(int64(packet.Time), 0).UTC()
	}

	var points []tsdb.Point
	for i := range packet.Values {
		name := fmt.Sprintf("%s_%s", packet.Plugin, packet.Values[i].Name)
		tags := make(map[string]string)
		fields := make(map[string]interface{})

		fields["value"] = packet.Values[i].Value

		if packet.Hostname != "" {
			tags["host"] = packet.Hostname
		}
		if packet.PluginInstance != "" {
			tags["instance"] = packet.PluginInstance
		}
		if packet.Type != "" {
			tags["type"] = packet.Type
		}
		if packet.TypeInstance != "" {
			tags["type_instance"] = packet.TypeInstance
		}
		p := tsdb.NewPoint(name, tags, fields, timestamp)

		points = append(points, p)
	}
	return points
}

// assert will panic with a given formatted message if the given condition is false.
func assert(condition bool, msg string, v ...interface{}) {
	if !condition {
		panic(fmt.Sprintf("assert failed: "+msg, v...))
	}
}
