package opentsdb // import "github.com/influxdata/influxdb/services/opentsdb"

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"io"
	"log"
	"net"
	"net/http"
	"net/textproto"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/tsdb"
)

// statistics gathered by the openTSDB package.
const (
	statHTTPConnectionsHandled   = "httpConnsHandled"
	statTelnetConnectionsActive  = "tlConnsActive"
	statTelnetConnectionsHandled = "tlConnsHandled"
	statTelnetPointsReceived     = "tlPointsRx"
	statTelnetBytesReceived      = "tlBytesRx"
	statTelnetReadError          = "tlReadErr"
	statTelnetBadLine            = "tlBadLine"
	statTelnetBadTime            = "tlBadTime"
	statTelnetBadTag             = "tlBadTag"
	statTelnetBadFloat           = "tlBadFloat"
	statBatchesTransmitted       = "batchesTx"
	statPointsTransmitted        = "pointsTx"
	statBatchesTransmitFail      = "batchesTxFail"
	statConnectionsActive        = "connsActive"
	statConnectionsHandled       = "connsHandled"
	statDroppedPointsInvalid     = "droppedPointsInvalid"
)

// Service manages the listener and handler for an HTTP endpoint.
type Service struct {
	ln     net.Listener  // main listener
	httpln *chanListener // http channel-based listener

	wg   sync.WaitGroup
	tls  bool
	cert string

	mu    sync.RWMutex
	ready bool          // Has the required database been created?
	done  chan struct{} // Is the service closing or closed?

	BindAddress     string
	Database        string
	RetentionPolicy string

	PointsWriter interface {
		WritePoints(database, retentionPolicy string, consistencyLevel models.ConsistencyLevel, points []models.Point) error
	}
	MetaClient interface {
		CreateDatabase(name string) (*meta.DatabaseInfo, error)
	}

	// Points received over the telnet protocol are batched.
	batchSize    int
	batchPending int
	batchTimeout time.Duration
	batcher      *tsdb.PointBatcher

	LogPointErrors bool
	Logger         *log.Logger

	stats       *Statistics
	defaultTags models.StatisticTags
}

// NewService returns a new instance of Service.
func NewService(c Config) (*Service, error) {
	// Use defaults where necessary.
	d := c.WithDefaults()

	s := &Service{
		tls:             d.TLSEnabled,
		cert:            d.Certificate,
		BindAddress:     d.BindAddress,
		Database:        d.Database,
		RetentionPolicy: d.RetentionPolicy,
		batchSize:       d.BatchSize,
		batchPending:    d.BatchPending,
		batchTimeout:    time.Duration(d.BatchTimeout),
		Logger:          log.New(os.Stderr, "[opentsdb] ", log.LstdFlags),
		LogPointErrors:  d.LogPointErrors,
		stats:           &Statistics{},
		defaultTags:     models.StatisticTags{"bind": d.BindAddress},
	}
	return s, nil
}

// Open starts the service
func (s *Service) Open() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.closed() {
		return nil // Already open.
	}
	s.done = make(chan struct{})

	s.Logger.Println("Starting OpenTSDB service")

	s.batcher = tsdb.NewPointBatcher(s.batchSize, s.batchPending, s.batchTimeout)
	s.batcher.Start()

	// Start processing batches.
	s.wg.Add(1)
	go func() { defer s.wg.Done(); s.processBatches(s.batcher) }()

	// Open listener.
	if s.tls {
		cert, err := tls.LoadX509KeyPair(s.cert, s.cert)
		if err != nil {
			return err
		}

		listener, err := tls.Listen("tcp", s.BindAddress, &tls.Config{
			Certificates: []tls.Certificate{cert},
		})
		if err != nil {
			return err
		}

		s.Logger.Println("Listening on TLS:", listener.Addr().String())
		s.ln = listener
	} else {
		listener, err := net.Listen("tcp", s.BindAddress)
		if err != nil {
			return err
		}

		s.Logger.Println("Listening on:", listener.Addr().String())
		s.ln = listener
	}
	s.httpln = newChanListener(s.ln.Addr())

	// Begin listening for connections.
	s.wg.Add(2)
	go func() { defer s.wg.Done(); s.serve() }()
	go func() { defer s.wg.Done(); s.serveHTTP() }()

	return nil
}

// Close closes the openTSDB service
func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed() {
		return nil // Already closed.
	}
	close(s.done)

	// Close the listeners.
	if err := s.ln.Close(); err != nil {
		return err
	}
	if err := s.httpln.Close(); err != nil {
		return err
	}

	s.wg.Wait()
	s.done = nil

	if s.batcher != nil {
		s.batcher.Stop()
	}

	return nil
}

// Closed returns true if the service is currently closed.
func (s *Service) Closed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed()
}

func (s *Service) closed() bool {
	select {
	case <-s.done:
		// Service is closing.
		return true
	default:
		return s.done == nil
	}
}

// createInternalStorage ensures that the required database has been created.
func (s *Service) createInternalStorage() error {
	s.mu.RLock()
	ready := s.ready
	s.mu.RUnlock()
	if ready {
		return nil
	}

	if _, err := s.MetaClient.CreateDatabase(s.Database); err != nil {
		return err
	}

	// The service is now ready.
	s.mu.Lock()
	s.ready = true
	s.mu.Unlock()
	return nil
}

// SetLogOutput sets the writer to which all logs are written. It must not be
// called after Open is called.
func (s *Service) SetLogOutput(w io.Writer) {
	s.Logger = log.New(w, "[opentsdb] ", log.LstdFlags)
}

// Statistics maintains statistics for the subscriber service.
type Statistics struct {
	HTTPConnectionsHandled   int64
	ActiveTelnetConnections  int64
	HandledTelnetConnections int64
	TelnetPointsReceived     int64
	TelnetBytesReceived      int64
	TelnetReadError          int64
	TelnetBadLine            int64
	TelnetBadTime            int64
	TelnetBadTag             int64
	TelnetBadFloat           int64
	BatchesTransmitted       int64
	PointsTransmitted        int64
	BatchesTransmitFail      int64
	ActiveConnections        int64
	HandledConnections       int64
	InvalidDroppedPoints     int64
}

// Statistics returns statistics for periodic monitoring.
func (s *Service) Statistics(tags map[string]string) []models.Statistic {
	return []models.Statistic{{
		Name: "opentsdb",
		Tags: s.defaultTags.Merge(tags),
		Values: map[string]interface{}{
			statHTTPConnectionsHandled:   atomic.LoadInt64(&s.stats.HTTPConnectionsHandled),
			statTelnetConnectionsActive:  atomic.LoadInt64(&s.stats.ActiveTelnetConnections),
			statTelnetConnectionsHandled: atomic.LoadInt64(&s.stats.HandledTelnetConnections),
			statTelnetPointsReceived:     atomic.LoadInt64(&s.stats.TelnetPointsReceived),
			statTelnetBytesReceived:      atomic.LoadInt64(&s.stats.TelnetBytesReceived),
			statTelnetReadError:          atomic.LoadInt64(&s.stats.TelnetReadError),
			statTelnetBadLine:            atomic.LoadInt64(&s.stats.TelnetBadLine),
			statTelnetBadTime:            atomic.LoadInt64(&s.stats.TelnetBadTime),
			statTelnetBadTag:             atomic.LoadInt64(&s.stats.TelnetBadTag),
			statTelnetBadFloat:           atomic.LoadInt64(&s.stats.TelnetBadFloat),
			statBatchesTransmitted:       atomic.LoadInt64(&s.stats.BatchesTransmitted),
			statPointsTransmitted:        atomic.LoadInt64(&s.stats.PointsTransmitted),
			statBatchesTransmitFail:      atomic.LoadInt64(&s.stats.BatchesTransmitFail),
			statConnectionsActive:        atomic.LoadInt64(&s.stats.ActiveConnections),
			statConnectionsHandled:       atomic.LoadInt64(&s.stats.HandledConnections),
			statDroppedPointsInvalid:     atomic.LoadInt64(&s.stats.InvalidDroppedPoints),
		},
	}}
}

// Err returns a channel for fatal errors that occur on the listener.
// func (s *Service) Err() <-chan error { return s.err }

// Addr returns the listener's address. Returns nil if listener is closed.
func (s *Service) Addr() net.Addr {
	if s.ln == nil {
		return nil
	}
	return s.ln.Addr()
}

// serve serves the handler from the listener.
func (s *Service) serve() {
	for {
		// Wait for next connection.
		conn, err := s.ln.Accept()
		if opErr, ok := err.(*net.OpError); ok && !opErr.Temporary() {
			s.Logger.Println("openTSDB TCP listener closed")
			return
		} else if err != nil {
			s.Logger.Println("error accepting openTSDB: ", err.Error())
			continue
		}

		// Handle connection in separate goroutine.
		go s.handleConn(conn)
	}
}

// handleConn processes conn. This is run in a separate goroutine.
func (s *Service) handleConn(conn net.Conn) {
	defer atomic.AddInt64(&s.stats.ActiveConnections, -1)
	atomic.AddInt64(&s.stats.ActiveConnections, 1)
	atomic.AddInt64(&s.stats.HandledConnections, 1)

	// Read header into buffer to check if it's HTTP.
	var buf bytes.Buffer
	r := bufio.NewReader(io.TeeReader(conn, &buf))

	// Attempt to parse connection as HTTP.
	_, err := http.ReadRequest(r)

	// Rebuild connection from buffer and remaining connection data.
	bufr := bufio.NewReader(io.MultiReader(&buf, conn))
	conn = &readerConn{Conn: conn, r: bufr}

	// If no HTTP parsing error occurred then process as HTTP.
	if err == nil {
		atomic.AddInt64(&s.stats.HTTPConnectionsHandled, 1)
		s.httpln.ch <- conn
		return
	}

	// Otherwise handle in telnet format.
	s.wg.Add(1)
	s.handleTelnetConn(conn)
	s.wg.Done()
}

// handleTelnetConn accepts OpenTSDB's telnet protocol.
// Each telnet command consists of a line of the form:
//   put sys.cpu.user 1356998400 42.5 host=webserver01 cpu=0
func (s *Service) handleTelnetConn(conn net.Conn) {
	defer conn.Close()
	defer atomic.AddInt64(&s.stats.ActiveTelnetConnections, -1)
	atomic.AddInt64(&s.stats.ActiveTelnetConnections, 1)
	atomic.AddInt64(&s.stats.HandledTelnetConnections, 1)

	// Get connection details.
	remoteAddr := conn.RemoteAddr().String()

	// Wrap connection in a text protocol reader.
	r := textproto.NewReader(bufio.NewReader(conn))
	for {
		line, err := r.ReadLine()
		if err != nil {
			if err != io.EOF {
				atomic.AddInt64(&s.stats.TelnetReadError, 1)
				s.Logger.Println("error reading from openTSDB connection", err.Error())
			}
			return
		}
		atomic.AddInt64(&s.stats.TelnetPointsReceived, 1)
		atomic.AddInt64(&s.stats.TelnetBytesReceived, int64(len(line)))

		inputStrs := strings.Fields(line)

		if len(inputStrs) == 1 && inputStrs[0] == "version" {
			conn.Write([]byte("InfluxDB TSDB proxy"))
			continue
		}

		if len(inputStrs) < 4 || inputStrs[0] != "put" {
			atomic.AddInt64(&s.stats.TelnetBadLine, 1)
			if s.LogPointErrors {
				s.Logger.Printf("malformed line '%s' from %s", line, remoteAddr)
			}
			continue
		}

		measurement := inputStrs[1]
		tsStr := inputStrs[2]
		valueStr := inputStrs[3]
		tagStrs := inputStrs[4:]

		var t time.Time
		ts, err := strconv.ParseInt(tsStr, 10, 64)
		if err != nil {
			atomic.AddInt64(&s.stats.TelnetBadTime, 1)
			if s.LogPointErrors {
				s.Logger.Printf("malformed time '%s' from %s", tsStr, remoteAddr)
			}
		}

		switch len(tsStr) {
		case 10:
			t = time.Unix(ts, 0)
			break
		case 13:
			t = time.Unix(ts/1000, (ts%1000)*1000)
			break
		default:
			atomic.AddInt64(&s.stats.TelnetBadTime, 1)
			if s.LogPointErrors {
				s.Logger.Printf("bad time '%s' must be 10 or 13 chars, from %s ", tsStr, remoteAddr)
			}
			continue
		}

		tags := make(map[string]string)
		for t := range tagStrs {
			parts := strings.SplitN(tagStrs[t], "=", 2)
			if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
				atomic.AddInt64(&s.stats.TelnetBadTag, 1)
				if s.LogPointErrors {
					s.Logger.Printf("malformed tag data '%v' from %s", tagStrs[t], remoteAddr)
				}
				continue
			}
			k := parts[0]

			tags[k] = parts[1]
		}

		fields := make(map[string]interface{})
		fv, err := strconv.ParseFloat(valueStr, 64)
		if err != nil {
			atomic.AddInt64(&s.stats.TelnetBadFloat, 1)
			if s.LogPointErrors {
				s.Logger.Printf("bad float '%s' from %s", valueStr, remoteAddr)
			}
			continue
		}
		fields["value"] = fv

		pt, err := models.NewPoint(measurement, models.NewTags(tags), fields, t)
		if err != nil {
			atomic.AddInt64(&s.stats.TelnetBadFloat, 1)
			if s.LogPointErrors {
				s.Logger.Printf("bad float '%s' from %s", valueStr, remoteAddr)
			}
			continue
		}
		s.batcher.In() <- pt
	}
}

// serveHTTP handles connections in HTTP format.
func (s *Service) serveHTTP() {
	handler := &Handler{
		Database:        s.Database,
		RetentionPolicy: s.RetentionPolicy,
		PointsWriter:    s.PointsWriter,
		Logger:          s.Logger,
		stats:           s.stats,
	}
	srv := &http.Server{Handler: handler}
	srv.Serve(s.httpln)
}

// processBatches continually drains the given batcher and writes the batches to the database.
func (s *Service) processBatches(batcher *tsdb.PointBatcher) {
	for {
		select {
		case <-s.done:
			return
		case batch := <-batcher.Out():
			// Will attempt to create database if not yet created.
			if err := s.createInternalStorage(); err != nil {
				s.Logger.Printf("Required database %s does not yet exist: %s", s.Database, err.Error())
				continue
			}

			if err := s.PointsWriter.WritePoints(s.Database, s.RetentionPolicy, models.ConsistencyLevelAny, batch); err == nil {
				atomic.AddInt64(&s.stats.BatchesTransmitted, 1)
				atomic.AddInt64(&s.stats.PointsTransmitted, int64(len(batch)))
			} else {
				s.Logger.Printf("failed to write point batch to database %q: %s", s.Database, err)
				atomic.AddInt64(&s.stats.BatchesTransmitFail, 1)
			}
		}
	}
}
