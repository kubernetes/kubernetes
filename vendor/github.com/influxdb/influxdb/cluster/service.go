package cluster

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
)

// MaxMessageSize defines how large a message can be before we reject it
const MaxMessageSize = 1024 * 1024 * 1024 // 1GB

// MuxHeader is the header byte used in the TCP mux.
const MuxHeader = 2

// Service processes data received over raw TCP connections.
type Service struct {
	mu sync.RWMutex

	wg      sync.WaitGroup
	closing chan struct{}

	Listener net.Listener

	MetaStore interface {
		ShardOwner(shardID uint64) (string, string, *meta.ShardGroupInfo)
	}

	TSDBStore interface {
		CreateShard(database, policy string, shardID uint64) error
		WriteToShard(shardID uint64, points []tsdb.Point) error
	}

	Logger *log.Logger
}

// NewService returns a new instance of Service.
func NewService(c Config) *Service {
	return &Service{
		closing: make(chan struct{}),
		Logger:  log.New(os.Stderr, "[tcp] ", log.LstdFlags),
	}
}

// Open opens the network listener and begins serving requests.
func (s *Service) Open() error {
	// Begin serving conections.
	s.wg.Add(1)
	go s.serve()

	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// serve accepts connections from the listener and handles them.
func (s *Service) serve() {
	defer s.wg.Done()

	for {
		// Check if the service is shutting down.
		select {
		case <-s.closing:
			return
		default:
		}

		// Accept the next connection.
		conn, err := s.Listener.Accept()
		if err != nil {
			if strings.Contains(err.Error(), "connection closed") {
				s.Logger.Printf("cluster service accept error: %s", err)
				return
			}
			s.Logger.Printf("accept error: %s", err)
			continue
		}

		// Delegate connection handling to a separate goroutine.
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			s.handleConn(conn)
		}()
	}
}

// Close shuts down the listener and waits for all connections to finish.
func (s *Service) Close() error {
	if s.Listener != nil {
		s.Listener.Close()
	}

	// Shut down all handlers.
	close(s.closing)
	// s.wg.Wait() // FIXME(benbjohnson)

	return nil
}

// handleConn services an individual TCP connection.
func (s *Service) handleConn(conn net.Conn) {
	// Ensure connection is closed when service is closed.
	closing := make(chan struct{})
	defer close(closing)
	go func() {
		select {
		case <-closing:
		case <-s.closing:
		}
		conn.Close()
	}()

	s.Logger.Printf("accept remote write connection from %v\n", conn.RemoteAddr())
	defer func() {
		s.Logger.Printf("close remote write connection from %v\n", conn.RemoteAddr())
	}()
	for {
		// Read type-length-value.
		typ, buf, err := ReadTLV(conn)
		if err != nil {
			if strings.HasSuffix(err.Error(), "EOF") {
				return
			}
			s.Logger.Printf("unable to read type-length-value %s", err)
			return
		}

		// Delegate message processing by type.
		switch typ {
		case writeShardRequestMessage:
			err := s.processWriteShardRequest(buf)
			if err != nil {
				s.Logger.Printf("process write shard error: %s", err)
			}
			s.writeShardResponse(conn, err)
		default:
			s.Logger.Printf("cluster service message type not found: %d", typ)
		}
	}
}

func (s *Service) processWriteShardRequest(buf []byte) error {
	// Build request
	var req WriteShardRequest
	if err := req.UnmarshalBinary(buf); err != nil {
		return err
	}

	err := s.TSDBStore.WriteToShard(req.ShardID(), req.Points())

	// We may have received a write for a shard that we don't have locally because the
	// sending node may have just created the shard (via the metastore) and the write
	// arrived before the local store could create the shard.  In this case, we need
	// to check the metastore to determine what database and retention policy this
	// shard should reside within.
	if err == tsdb.ErrShardNotFound {

		// Query the metastore for the owner of this shard
		database, retentionPolicy, sgi := s.MetaStore.ShardOwner(req.ShardID())
		if sgi == nil {
			// If we can't find it, then we need to drop this request
			// as it is no longer valid.  This could happen if writes were queued via
			// hinted handoff and delivered after a shard group was deleted.
			s.Logger.Printf("drop write request: shard=%d", req.ShardID())
			return nil
		}

		err = s.TSDBStore.CreateShard(database, retentionPolicy, req.ShardID())
		if err != nil {
			return err
		}
		return s.TSDBStore.WriteToShard(req.ShardID(), req.Points())
	}

	if err != nil {
		return fmt.Errorf("write shard %d: %s", req.ShardID(), err)
	}

	return nil
}

func (s *Service) writeShardResponse(w io.Writer, e error) {
	// Build response.
	var resp WriteShardResponse
	if e != nil {
		resp.SetCode(1)
		resp.SetMessage(e.Error())
	} else {
		resp.SetCode(0)
	}

	// Marshal response to binary.
	buf, err := resp.MarshalBinary()
	if err != nil {
		s.Logger.Printf("error marshalling shard response: %s", err)
		return
	}

	// Write to connection.
	if err := WriteTLV(w, writeShardResponseMessage, buf); err != nil {
		s.Logger.Printf("write shard response error: %s", err)
	}
}

// ReadTLV reads a type-length-value record from r.
func ReadTLV(r io.Reader) (byte, []byte, error) {
	var typ [1]byte
	if _, err := io.ReadFull(r, typ[:]); err != nil {
		return 0, nil, fmt.Errorf("read message type: %s", err)
	}

	// Read the size of the message.
	var sz int64
	if err := binary.Read(r, binary.BigEndian, &sz); err != nil {
		return 0, nil, fmt.Errorf("read message size: %s", err)
	}

	if sz == 0 {
		return 0, nil, fmt.Errorf("invalid message size: %d", sz)
	}

	if sz >= MaxMessageSize {
		return 0, nil, fmt.Errorf("max message size of %d exceeded: %d", MaxMessageSize, sz)
	}

	// Read the value.
	buf := make([]byte, sz)
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, nil, fmt.Errorf("read message value: %s", err)
	}

	return typ[0], buf, nil
}

// WriteTLV writes a type-length-value record to w.
func WriteTLV(w io.Writer, typ byte, buf []byte) error {
	if _, err := w.Write([]byte{typ}); err != nil {
		return fmt.Errorf("write message type: %s", err)
	}

	// Write the size of the message.
	if err := binary.Write(w, binary.BigEndian, int64(len(buf))); err != nil {
		return fmt.Errorf("write message size: %s", err)
	}

	// Write the value.
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("write message value: %s", err)
	}

	return nil
}
