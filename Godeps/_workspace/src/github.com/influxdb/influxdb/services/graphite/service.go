package graphite

import (
	"bufio"
	"expvar"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/monitor"
	"github.com/influxdb/influxdb/tsdb"
)

const (
	udpBufferSize     = 65536
	leaderWaitTimeout = 30 * time.Second
)

// statistics gathered by the graphite package.
const (
	statPointsReceived      = "pointsRx"
	statBytesReceived       = "bytesRx"
	statPointsParseFail     = "pointsParseFail"
	statPointsNaNFail       = "pointsNaNFail"
	statPointsUnsupported   = "pointsUnsupportedFail"
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
	mu sync.Mutex

	bindAddress      string
	database         string
	protocol         string
	batchSize        int
	batchPending     int
	batchTimeout     time.Duration
	consistencyLevel cluster.ConsistencyLevel
	udpReadBuffer    int

	batcher *tsdb.PointBatcher
	parser  *Parser

	logger           *log.Logger
	statMap          *expvar.Map
	tcpConnectionsMu sync.Mutex
	tcpConnections   map[string]*tcpConnection

	ln      net.Listener
	addr    net.Addr
	udpConn *net.UDPConn

	wg   sync.WaitGroup
	done chan struct{}

	Monitor interface {
		RegisterDiagnosticsClient(name string, client monitor.DiagsClient)
		DeregisterDiagnosticsClient(name string)
	}
	PointsWriter interface {
		WritePoints(p *cluster.WritePointsRequest) error
	}
	MetaStore interface {
		WaitForLeader(d time.Duration) error
		CreateDatabaseIfNotExists(name string) (*meta.DatabaseInfo, error)
	}
}

// NewService returns an instance of the Graphite service.
func NewService(c Config) (*Service, error) {
	// Use defaults where necessary.
	d := c.WithDefaults()

	s := Service{
		bindAddress:    d.BindAddress,
		database:       d.Database,
		protocol:       d.Protocol,
		batchSize:      d.BatchSize,
		batchPending:   d.BatchPending,
		udpReadBuffer:  d.UDPReadBuffer,
		batchTimeout:   time.Duration(d.BatchTimeout),
		logger:         log.New(os.Stderr, "[graphite] ", log.LstdFlags),
		tcpConnections: make(map[string]*tcpConnection),
		done:           make(chan struct{}),
	}

	consistencyLevel, err := cluster.ParseConsistencyLevel(d.ConsistencyLevel)
	if err != nil {
		return nil, err
	}
	s.consistencyLevel = consistencyLevel

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

	s.logger.Printf("Starting graphite service, batch size %d, batch timeout %s", s.batchSize, s.batchTimeout)

	// Configure expvar monitoring. It's OK to do this even if the service fails to open and
	// should be done before any data could arrive for the service.
	key := strings.Join([]string{"graphite", s.protocol, s.bindAddress}, ":")
	tags := map[string]string{"proto": s.protocol, "bind": s.bindAddress}
	s.statMap = influxdb.NewStatistics(key, "graphite", tags)

	// Register diagnostics if a Monitor service is available.
	if s.Monitor != nil {
		s.Monitor.RegisterDiagnosticsClient(key, s)
	}

	if err := s.MetaStore.WaitForLeader(leaderWaitTimeout); err != nil {
		s.logger.Printf("Failed to detect a cluster leader: %s", err.Error())
		return err
	}

	if _, err := s.MetaStore.CreateDatabaseIfNotExists(s.database); err != nil {
		s.logger.Printf("Failed to ensure target database %s exists: %s", s.database, err.Error())
		return err
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
	close(s.done)
	s.wg.Wait()
	s.done = nil

	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.logger = l
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
	defer s.statMap.Add(statConnectionsActive, -1)
	defer s.untrackConnection(conn)
	s.statMap.Add(statConnectionsActive, 1)
	s.statMap.Add(statConnectionsHandled, 1)
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

		s.statMap.Add(statPointsReceived, 1)
		s.statMap.Add(statBytesReceived, int64(len(buf)))
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
			s.statMap.Add(statPointsReceived, int64(len(lines)))
			s.statMap.Add(statBytesReceived, int64(n))
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
		case *UnsupposedValueError:
			// Graphite ignores NaN values with no error.
			if math.IsNaN(err.Value) {
				s.statMap.Add(statPointsNaNFail, 1)
				return
			}
		}
		s.logger.Printf("unable to parse line: %s: %s", line, err)
		s.statMap.Add(statPointsParseFail, 1)
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
			if err := s.PointsWriter.WritePoints(&cluster.WritePointsRequest{
				Database:         s.database,
				RetentionPolicy:  "",
				ConsistencyLevel: s.consistencyLevel,
				Points:           batch,
			}); err == nil {
				s.statMap.Add(statBatchesTransmitted, 1)
				s.statMap.Add(statPointsTransmitted, int64(len(batch)))
			} else {
				s.logger.Printf("failed to write point batch to database %q: %s", s.database, err)
				s.statMap.Add(statBatchesTransmitFail, 1)
			}

		case <-s.done:
			return
		}
	}
}

// Diagnostics returns diagnostics of the graphite service.
func (s *Service) Diagnostics() (*monitor.Diagnostic, error) {
	s.tcpConnectionsMu.Lock()
	defer s.tcpConnectionsMu.Unlock()

	d := &monitor.Diagnostic{
		Columns: []string{"local", "remote", "connect time"},
		Rows:    make([][]interface{}, 0, len(s.tcpConnections)),
	}
	for _, v := range s.tcpConnections {
		_ = v
		d.Rows = append(d.Rows, []interface{}{v.conn.LocalAddr().String(), v.conn.RemoteAddr().String(), v.connectTime})
	}
	return d, nil
}
