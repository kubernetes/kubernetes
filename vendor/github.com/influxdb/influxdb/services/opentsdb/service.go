package opentsdb

import (
	"bufio"
	"bytes"
	"io"
	"log"
	"net"
	"net/http"
	"net/textproto"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
)

const leaderWaitTimeout = 30 * time.Second

// Service manages the listener and handler for an HTTP endpoint.
type Service struct {
	ln     net.Listener  // main listener
	httpln *chanListener // http channel-based listener

	wg  sync.WaitGroup
	err chan error

	BindAddress      string
	Database         string
	RetentionPolicy  string
	ConsistencyLevel cluster.ConsistencyLevel

	PointsWriter interface {
		WritePoints(p *cluster.WritePointsRequest) error
	}
	MetaStore interface {
		WaitForLeader(d time.Duration) error
		CreateDatabaseIfNotExists(name string) (*meta.DatabaseInfo, error)
	}

	Logger *log.Logger
}

// NewService returns a new instance of Service.
func NewService(c Config) (*Service, error) {
	consistencyLevel, err := cluster.ParseConsistencyLevel(c.ConsistencyLevel)
	if err != nil {
		return nil, err
	}

	s := &Service{
		err:              make(chan error),
		BindAddress:      c.BindAddress,
		Database:         c.Database,
		RetentionPolicy:  c.RetentionPolicy,
		ConsistencyLevel: consistencyLevel,
		Logger:           log.New(os.Stderr, "[opentsdb] ", log.LstdFlags),
	}
	return s, nil
}

// Open starts the service
func (s *Service) Open() error {
	if err := s.MetaStore.WaitForLeader(leaderWaitTimeout); err != nil {
		s.Logger.Printf("failed to detect a cluster leader: %s", err.Error())
		return err
	}

	if _, err := s.MetaStore.CreateDatabaseIfNotExists(s.Database); err != nil {
		s.Logger.Printf("failed to ensure target database %s exists: %s", s.Database, err.Error())
		return err
	}
	s.Logger.Printf("ensured target database %s exists", s.Database)

	// Open listener.
	ln, err := net.Listen("tcp", s.BindAddress)
	if err != nil {
		return err
	}
	s.ln = ln
	s.httpln = newChanListener(ln.Addr())

	s.Logger.Println("listening on:", ln.Addr().String())

	// Begin listening for connections.
	s.wg.Add(2)
	go s.serveHTTP()
	go s.serve()

	return nil
}

// Close closes the underlying listener.
func (s *Service) Close() error {
	if s.ln != nil {
		return s.ln.Close()
	}

	s.wg.Wait()
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) { s.Logger = l }

// Err returns a channel for fatal errors that occur on the listener.
func (s *Service) Err() <-chan error { return s.err }

// Addr returns the listener's address. Returns nil if listener is closed.
func (s *Service) Addr() net.Addr {
	if s.ln == nil {
		return nil
	}
	return s.ln.Addr()
}

// serve serves the handler from the listener.
func (s *Service) serve() {
	defer s.wg.Done()

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
		s.httpln.ch <- conn
		return
	}

	// Otherwise handle in telnet format.
	s.wg.Add(1)
	s.handleTelnetConn(conn)
}

// handleTelnetConn accepts OpenTSDB's telnet protocol.
// Each telnet command consists of a line of the form:
//   put sys.cpu.user 1356998400 42.5 host=webserver01 cpu=0
func (s *Service) handleTelnetConn(conn net.Conn) {
	defer conn.Close()
	defer s.wg.Done()

	// Wrap connection in a text protocol reader.
	r := textproto.NewReader(bufio.NewReader(conn))
	for {
		line, err := r.ReadLine()
		if err != nil {
			s.Logger.Println("error reading from openTSDB connection", err.Error())
			return
		}

		inputStrs := strings.Fields(line)

		if len(inputStrs) == 1 && inputStrs[0] == "version" {
			conn.Write([]byte("InfluxDB TSDB proxy"))
			continue
		}

		if len(inputStrs) < 4 || inputStrs[0] != "put" {
			s.Logger.Println("TSDBServer: malformed line, skipping: ", line)
			continue
		}

		measurement := inputStrs[1]
		tsStr := inputStrs[2]
		valueStr := inputStrs[3]
		tagStrs := inputStrs[4:]

		var t time.Time
		ts, err := strconv.ParseInt(tsStr, 10, 64)
		if err != nil {
			s.Logger.Println("TSDBServer: malformed time, skipping: ", tsStr)
		}

		switch len(tsStr) {
		case 10:
			t = time.Unix(ts, 0)
			break
		case 13:
			t = time.Unix(ts/1000, (ts%1000)*1000)
			break
		default:
			s.Logger.Println("TSDBServer: time must be 10 or 13 chars, skipping: ", tsStr)
			continue
		}

		tags := make(map[string]string)
		for t := range tagStrs {
			parts := strings.SplitN(tagStrs[t], "=", 2)
			if len(parts) != 2 {
				s.Logger.Println("TSDBServer: malformed tag data", tagStrs[t])
				continue
			}
			k := parts[0]

			tags[k] = parts[1]
		}

		fields := make(map[string]interface{})
		fields["value"], err = strconv.ParseFloat(valueStr, 64)
		if err != nil {
			s.Logger.Println("TSDBServer: could not parse value as float: ", valueStr)
			continue
		}

		p := tsdb.NewPoint(measurement, tags, fields, t)
		if err := s.PointsWriter.WritePoints(&cluster.WritePointsRequest{
			Database:         s.Database,
			RetentionPolicy:  s.RetentionPolicy,
			ConsistencyLevel: s.ConsistencyLevel,
			Points:           []tsdb.Point{p},
		}); err != nil {
			s.Logger.Println("TSDB cannot write data: ", err)
			continue
		}
	}
}

// serveHTTP handles connections in HTTP format.
func (s *Service) serveHTTP() {
	srv := &http.Server{Handler: &Handler{
		Database:         s.Database,
		RetentionPolicy:  s.RetentionPolicy,
		ConsistencyLevel: s.ConsistencyLevel,
		PointsWriter:     s.PointsWriter,
		Logger:           s.Logger,
	}}
	srv.Serve(s.httpln)
}
