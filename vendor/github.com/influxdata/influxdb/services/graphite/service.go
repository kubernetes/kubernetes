package graphite // import "github.com/influxdata/influxdb/services/graphite"

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/monitor/diagnostics"
	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/tsdb"
)

const udpBufferSize = 65536

// statistics gathered by the graphite package.
const (
	statPointsReceived      = "pointsRx"
	statBytesReceived       = "bytesRx"
	statPointsParseFail     = "pointsParseFail"
	statPointsNaNFail       = "pointsNaNFail"
	statBatchesTransmitted  = "batchesTx"
	statPointsTransmitted   = "pointsTx"
	statBatchesTransmitFail = "batchesTxFail"
	statConnectionsActive   = "connsActive"
	statConnectionsHandled  = "connsHandled"
)

type tcpConnection struct {
	conn        net.Conn
	connectTime time.Time
}

func (c *tcpConnection) Close() {
	c.conn.Close()
}

// Service represents a Graphite service.
type Service struct {
	bindAddress     string
	database        string
	retentionPolicy string
	protocol        string
	batchSize       int
	batchPending    int
	batchTimeout    time.Duration
	udpReadBuffer   int

	batcher *tsdb.PointBatcher
	parser  *Parser

	logger      *log.Logger
	stats       *Statistics
	defaultTags models.StatisticTags

	tcpConnectionsMu sync.Mutex
	tcpConnections   map[string]*tcpConnection
	diagsKey         string

	ln      net.Listener
	addr    net.Addr
	udpConn *net.UDPConn

	wg sync.WaitGroup

	mu    sync.RWMutex
	ready bool          // Has the required database been created?
	done  chan struct{} // Is the service closing or closed?

	Monitor interface {
		RegisterDiagnosticsClient(name string, client diagnostics.Client)
		DeregisterDiagnosticsClient(name string)
	}
	PointsWriter interface {
		WritePoints(database, retentionPolicy string, consistencyLevel models.ConsistencyLevel, points []models.Point) error
	}
	MetaClient interface {
		CreateDatabase(name string) (*meta.DatabaseInfo, error)
		CreateDatabaseWithRetentionPolicy(name string, spec *meta.RetentionPolicySpec) (*meta.DatabaseInfo, error)
		CreateRetentionPolicy(database string, spec *meta.RetentionPolicySpec) (*meta.RetentionPolicyInfo, error)
		Database(name string) *meta.DatabaseInfo
		RetentionPolicy(database, name string) (*meta.RetentionPolicyInfo, error)
	}
}

// NewService returns an instance of the Graphite service.
func NewService(c Config) (*Service, error) {
	// Use defaults where necessary.
	d := c.WithDefaults()

	s := Service{
		bindAddress:     d.BindAddress,
		database:        d.Database,
		retentionPolicy: d.RetentionPolicy,
		protocol:        d.Protocol,
		batchSize:       d.BatchSize,
		batchPending:    d.BatchPending,
		udpReadBuffer:   d.UDPReadBuffer,
		batchTimeout:    time.Duration(d.BatchTimeout),
		logger:          log.New(os.Stderr, fmt.Sprintf("[graphite] %s ", d.BindAddress), log.LstdFlags),
		stats:           &Statistics{},
		defaultTags:     models.StatisticTags{"proto": d.Protocol, "bind": d.BindAddress},
		tcpConnections:  make(map[string]*tcpConnection),
		diagsKey:        strings.Join([]string{"graphite", d.Protocol, d.BindAddress}, ":"),
	}

	parser, err := NewParserWithOptions(Options{
		Templates:   d.Templates,
		DefaultTags: d.DefaultTags(),
		Separator:   d.Separator})

	if err != nil {
		return nil, err
	}
	s.parser = parser

	return &s, nil
}

// Open starts the Graphite input processing data.
func (s *Service) Open() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.closed() {
		return nil // Already open.
	}
	s.done = make(chan struct{})

	s.logger.Printf("Starting graphite service, batch size %d, batch timeout %s", s.batchSize, s.batchTimeout)

	// Register diagnostics if a Monitor service is available.
	if s.Monitor != nil {
		s.Monitor.RegisterDiagnosticsClient(s.diagsKey, s)
	}

	s.batcher = tsdb.NewPointBatcher(s.batchSize, s.batchPending, s.batchTimeout)
	s.batcher.Start()

	// Start processing batches.
	s.wg.Add(1)
	go s.processBatches(s.batcher)

	var err error
	if strings.ToLower(s.protocol) == "tcp" {
		s.addr, err = s.openTCPServer()
	} else if strings.ToLower(s.protocol) == "udp" {
		s.addr, err = s.openUDPServer()
	} else {
		return fmt.Errorf("unrecognized Graphite input protocol %s", s.protocol)
	}
	if err != nil {
		return err
	}

	s.logger.Printf("Listening on %s: %s", strings.ToUpper(s.protocol), s.addr.String())
	return nil
}
func (s *Service) closeAllConnections() {
	s.tcpConnectionsMu.Lock()
	defer s.tcpConnectionsMu.Unlock()
	for _, c := range s.tcpConnections {
		c.Close()
	}
}

// Close stops all data processing on the Graphite input.
func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed() {
		return nil // Already closed.
	}
	close(s.done)

	s.closeAllConnections()

	if s.ln != nil {
		s.ln.Close()
	}
	if s.udpConn != nil {
		s.udpConn.Close()
	}

	if s.batcher != nil {
		s.batcher.Stop()
	}

	if s.Monitor != nil {
		s.Monitor.DeregisterDiagnosticsClient(s.diagsKey)
	}

	s.wg.Wait()
	s.done = nil

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
	}
	return s.done == nil
}

// createInternalStorage ensures that the required database has been created.
func (s *Service) createInternalStorage() error {
	s.mu.RLock()
	ready := s.ready
	s.mu.RUnlock()
	if ready {
		return nil
	}

	if db := s.MetaClient.Database(s.database); db != nil {
		if rp, _ := s.MetaClient.RetentionPolicy(s.database, s.retentionPolicy); rp == nil {
			spec := meta.RetentionPolicySpec{Name: s.retentionPolicy}
			if _, err := s.MetaClient.CreateRetentionPolicy(s.database, &spec); err != nil {
				return err
			}
		}
	} else {
		spec := meta.RetentionPolicySpec{Name: s.retentionPolicy}
		if _, err := s.MetaClient.CreateDatabaseWithRetentionPolicy(s.database, &spec); err != nil {
			return err
		}
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
	s.logger = log.New(w, "[graphite] ", log.LstdFlags)
}

// Statistics maintains statistics for the graphite service.
type Statistics struct {
	PointsReceived      int64
	BytesReceived       int64
	PointsParseFail     int64
	PointsNaNFail       int64
	BatchesTransmitted  int64
	PointsTransmitted   int64
	BatchesTransmitFail int64
	ActiveConnections   int64
	HandledConnections  int64
}

// Statistics returns statistics for periodic monitoring.
func (s *Service) Statistics(tags map[string]string) []models.Statistic {
	return []models.Statistic{{
		Name: "graphite",
		Tags: s.defaultTags.Merge(tags),
		Values: map[string]interface{}{
			statPointsReceived:      atomic.LoadInt64(&s.stats.PointsReceived),
			statBytesReceived:       atomic.LoadInt64(&s.stats.BytesReceived),
			statPointsParseFail:     atomic.LoadInt64(&s.stats.PointsParseFail),
			statPointsNaNFail:       atomic.LoadInt64(&s.stats.PointsNaNFail),
			statBatchesTransmitted:  atomic.LoadInt64(&s.stats.BatchesTransmitted),
			statPointsTransmitted:   atomic.LoadInt64(&s.stats.PointsTransmitted),
			statBatchesTransmitFail: atomic.LoadInt64(&s.stats.BatchesTransmitFail),
			statConnectionsActive:   atomic.LoadInt64(&s.stats.ActiveConnections),
			statConnectionsHandled:  atomic.LoadInt64(&s.stats.HandledConnections),
		},
	}}
}

// Addr returns the address the Service binds to.
func (s *Service) Addr() net.Addr {
	return s.addr
}

// openTCPServer opens the Graphite input in TCP mode and starts processing data.
func (s *Service) openTCPServer() (net.Addr, error) {
	ln, err := net.Listen("tcp", s.bindAddress)
	if err != nil {
		return nil, err
	}
	s.ln = ln

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		for {
			conn, err := s.ln.Accept()
			if opErr, ok := err.(*net.OpError); ok && !opErr.Temporary() {
				s.logger.Println("graphite TCP listener closed")
				return
			}
			if err != nil {
				s.logger.Println("error accepting TCP connection", err.Error())
				continue
			}

			s.wg.Add(1)
			go s.handleTCPConnection(conn)
		}
	}()
	return ln.Addr(), nil
}

// handleTCPConnection services an individual TCP connection for the Graphite input.
func (s *Service) handleTCPConnection(conn net.Conn) {
	defer s.wg.Done()
	defer conn.Close()
	defer atomic.AddInt64(&s.stats.ActiveConnections, -1)
	defer s.untrackConnection(conn)
	atomic.AddInt64(&s.stats.ActiveConnections, 1)
	atomic.AddInt64(&s.stats.HandledConnections, 1)
	s.trackConnection(conn)

	reader := bufio.NewReader(conn)

	for {
		// Read up to the next newline.
		buf, err := reader.ReadBytes('\n')
		if err != nil {
			return
		}

		// Trim the buffer, even though there should be no padding
		line := strings.TrimSpace(string(buf))

		atomic.AddInt64(&s.stats.PointsReceived, 1)
		atomic.AddInt64(&s.stats.BytesReceived, int64(len(buf)))
		s.handleLine(line)
	}
}

func (s *Service) trackConnection(c net.Conn) {
	s.tcpConnectionsMu.Lock()
	defer s.tcpConnectionsMu.Unlock()
	s.tcpConnections[c.RemoteAddr().String()] = &tcpConnection{
		conn:        c,
		connectTime: time.Now().UTC(),
	}
}
func (s *Service) untrackConnection(c net.Conn) {
	s.tcpConnectionsMu.Lock()
	defer s.tcpConnectionsMu.Unlock()
	delete(s.tcpConnections, c.RemoteAddr().String())
}

// openUDPServer opens the Graphite input in UDP mode and starts processing incoming data.
func (s *Service) openUDPServer() (net.Addr, error) {
	addr, err := net.ResolveUDPAddr("udp", s.bindAddress)
	if err != nil {
		return nil, err
	}

	s.udpConn, err = net.ListenUDP("udp", addr)
	if err != nil {
		return nil, err
	}

	if s.udpReadBuffer != 0 {
		err = s.udpConn.SetReadBuffer(s.udpReadBuffer)
		if err != nil {
			return nil, fmt.Errorf("unable to set UDP read buffer to %d: %s",
				s.udpReadBuffer, err)
		}
	}

	buf := make([]byte, udpBufferSize)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		for {
			n, _, err := s.udpConn.ReadFromUDP(buf)
			if err != nil {
				s.udpConn.Close()
				return
			}

			lines := strings.Split(string(buf[:n]), "\n")
			for _, line := range lines {
				s.handleLine(line)
			}
			atomic.AddInt64(&s.stats.PointsReceived, int64(len(lines)))
			atomic.AddInt64(&s.stats.BytesReceived, int64(n))
		}
	}()
	return s.udpConn.LocalAddr(), nil
}

func (s *Service) handleLine(line string) {
	if line == "" {
		return
	}

	// Parse it.
	point, err := s.parser.Parse(line)
	if err != nil {
		switch err := err.(type) {
		case *UnsupportedValueError:
			// Graphite ignores NaN values with no error.
			if math.IsNaN(err.Value) {
				atomic.AddInt64(&s.stats.PointsNaNFail, 1)
				return
			}
		}
		s.logger.Printf("unable to parse line: %s: %s", line, err)
		atomic.AddInt64(&s.stats.PointsParseFail, 1)
		return
	}

	s.batcher.In() <- point
}

// processBatches continually drains the given batcher and writes the batches to the database.
func (s *Service) processBatches(batcher *tsdb.PointBatcher) {
	defer s.wg.Done()
	for {
		select {
		case batch := <-batcher.Out():
			// Will attempt to create database if not yet created.
			if err := s.createInternalStorage(); err != nil {
				s.logger.Printf("Required database or retention policy do not yet exist: %s", err.Error())
				continue
			}

			if err := s.PointsWriter.WritePoints(s.database, s.retentionPolicy, models.ConsistencyLevelAny, batch); err == nil {
				atomic.AddInt64(&s.stats.BatchesTransmitted, 1)
				atomic.AddInt64(&s.stats.PointsTransmitted, int64(len(batch)))
			} else {
				s.logger.Printf("failed to write point batch to database %q: %s", s.database, err)
				atomic.AddInt64(&s.stats.BatchesTransmitFail, 1)
			}

		case <-s.done:
			return
		}
	}
}

// Diagnostics returns diagnostics of the graphite service.
func (s *Service) Diagnostics() (*diagnostics.Diagnostics, error) {
	s.tcpConnectionsMu.Lock()
	defer s.tcpConnectionsMu.Unlock()

	d := &diagnostics.Diagnostics{
		Columns: []string{"local", "remote", "connect time"},
		Rows:    make([][]interface{}, 0, len(s.tcpConnections)),
	}
	for _, v := range s.tcpConnections {
		d.Rows = append(d.Rows, []interface{}{v.conn.LocalAddr().String(), v.conn.RemoteAddr().String(), v.connectTime})
	}
	return d, nil
}
