package snapshotter

import (
	"encoding"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/influxdb/influxdb/snapshot"
	"github.com/influxdb/influxdb/tsdb"
)

// MuxHeader is the header byte used for the TCP muxer.
const MuxHeader = 3

// Service manages the listener for the snapshot endpoint.
type Service struct {
	wg  sync.WaitGroup
	err chan error

	MetaStore interface {
		encoding.BinaryMarshaler
	}

	TSDBStore *tsdb.Store

	Listener net.Listener
	Logger   *log.Logger
}

// NewService returns a new instance of Service.
func NewService() *Service {
	return &Service{
		err:    make(chan error),
		Logger: log.New(os.Stderr, "[snapshot] ", log.LstdFlags),
	}
}

// Open starts the service.
func (s *Service) Open() error {
	s.Logger.Println("Starting snapshot service")

	s.wg.Add(1)
	go s.serve()
	return nil
}

// Close implements the Service interface.
func (s *Service) Close() error {
	if s.Listener != nil {
		s.Listener.Close()
	}
	s.wg.Wait()
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// Err returns a channel for fatal out-of-band errors.
func (s *Service) Err() <-chan error { return s.err }

// serve serves snapshot requests from the listener.
func (s *Service) serve() {
	defer s.wg.Done()

	for {
		// Wait for next connection.
		conn, err := s.Listener.Accept()
		if err != nil && strings.Contains(err.Error(), "connection closed") {
			s.Logger.Println("snapshot listener closed")
			return
		} else if err != nil {
			s.Logger.Println("error accepting snapshot request: ", err.Error())
			continue
		}

		// Handle connection in separate goroutine.
		s.wg.Add(1)
		go func(conn net.Conn) {
			defer s.wg.Done()
			defer conn.Close()
			if err := s.handleConn(conn); err != nil {
				s.Logger.Println(err)
			}
		}(conn)
	}
}

// handleConn processes conn. This is run in a separate goroutine.
func (s *Service) handleConn(conn net.Conn) error {
	// Read manifest from connection.
	m, err := s.readManifest(conn)
	if err != nil {
		return fmt.Errorf("read manifest: %s", err)
	}

	// Write snapshot to connection.
	if err := s.writeSnapshot(conn, m); err != nil {
		return fmt.Errorf("write snapshot: %s", err)
	}

	return nil
}

// readManifest reads the manifest size and contents from conn.
// Unmarshals the bytes and returns a manifest object.
func (s *Service) readManifest(conn net.Conn) (snapshot.Manifest, error) {
	var m snapshot.Manifest
	if err := json.NewDecoder(conn).Decode(&m); err != nil {
		return m, err
	}
	return m, nil
}

// writeSnapshot creates a snapshot writer, trims the manifest, and writes to conn.
func (s *Service) writeSnapshot(conn net.Conn, prev snapshot.Manifest) error {
	// Retrieve and serialize the current meta data.
	buf, err := s.MetaStore.MarshalBinary()
	if err != nil {
		return fmt.Errorf("marshal meta: %s", err)
	}

	// Build a snapshot writer.
	sw, err := tsdb.NewSnapshotWriter(buf, s.TSDBStore)
	if err != nil {
		return fmt.Errorf("create snapshot writer: %s", err)
	}

	// Trim old files from snapshot.
	sw.Manifest = sw.Manifest.Diff(&prev)

	// Write snapshot out to connection.
	if _, err := sw.WriteTo(conn); err != nil {
		return fmt.Errorf("write to: %s", err)
	}

	return nil
}
