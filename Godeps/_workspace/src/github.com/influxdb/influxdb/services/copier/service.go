package copier

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/gogo/protobuf/proto"
	"github.com/influxdb/influxdb/services/copier/internal"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/tsdb"
)

//go:generate protoc --gogo_out=. internal/internal.proto

// MuxHeader is the header byte used for the TCP muxer.
const MuxHeader = 6

// Service manages the listener for the endpoint.
type Service struct {
	wg  sync.WaitGroup
	err chan error

	TSDBStore interface {
		Shard(id uint64) *tsdb.Shard
	}

	Listener net.Listener
	Logger   *log.Logger
}

// NewService returns a new instance of Service.
func NewService() *Service {
	return &Service{
		err:    make(chan error),
		Logger: log.New(os.Stderr, "[copier] ", log.LstdFlags),
	}
}

// Open starts the service.
func (s *Service) Open() error {
	s.Logger.Println("Starting copier service")

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

// serve serves shard copy requests from the listener.
func (s *Service) serve() {
	defer s.wg.Done()

	for {
		// Wait for next connection.
		conn, err := s.Listener.Accept()
		if err != nil && strings.Contains(err.Error(), "connection closed") {
			s.Logger.Println("copier listener closed")
			return
		} else if err != nil {
			s.Logger.Println("error accepting copier request: ", err.Error())
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
	// Read request from connection.
	req, err := s.readRequest(conn)
	if err != nil {
		return fmt.Errorf("read request: %s", err)
	}

	// Retrieve shard.
	sh := s.TSDBStore.Shard(req.GetShardID())

	// Return error response if the shard doesn't exist.
	if sh == nil {
		if err := s.writeResponse(conn, &internal.Response{
			Error: proto.String(fmt.Sprintf("shard not found: id=%d", req.GetShardID())),
		}); err != nil {
			return fmt.Errorf("write error response: %s", err)
		}
		return nil
	}

	// Write successful response.
	if err := s.writeResponse(conn, &internal.Response{}); err != nil {
		return fmt.Errorf("write response: %s", err)
	}

	// Write shard to response.
	if _, err := sh.WriteTo(conn); err != nil {
		return fmt.Errorf("write shard: %s", err)
	}

	return nil
}

// readRequest reads and unmarshals a Request from r.
func (s *Service) readRequest(r io.Reader) (*internal.Request, error) {
	// Read request length.
	var n uint32
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, fmt.Errorf("read request length: %s", err)
	}

	// Read body.
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("read request: %s", err)
	}

	// Unmarshal request.
	req := &internal.Request{}
	if err := proto.Unmarshal(buf, req); err != nil {
		return nil, fmt.Errorf("unmarshal request: %s", err)
	}

	return req, nil
}

// writeResponse marshals and writes a Response to w.
func (s *Service) writeResponse(w io.Writer, resp *internal.Response) error {
	// Marshal the response to a byte slice.
	buf, err := proto.Marshal(resp)
	if err != nil {
		return fmt.Errorf("marshal error: %s", err)
	}

	// Write response length to writer.
	if err := binary.Write(w, binary.BigEndian, uint32(len(buf))); err != nil {
		return fmt.Errorf("write response length error: %s", err)
	}

	// Write body to writer.
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("write body error: %s", err)
	}

	return nil
}

// Client represents a client for connecting remotely to a copier service.
type Client struct {
	host string
}

// NewClient return a new instance of Client.
func NewClient(host string) *Client {
	return &Client{
		host: host,
	}
}

// ShardReader returns a reader for streaming shard data.
// Returned ReadCloser must be closed by the caller.
func (c *Client) ShardReader(id uint64) (io.ReadCloser, error) {
	// Connect to remote server.
	conn, err := tcp.Dial("tcp", c.host, MuxHeader)
	if err != nil {
		return nil, err
	}

	// Send request to server.
	if err := c.writeRequest(conn, &internal.Request{ShardID: proto.Uint64(id)}); err != nil {
		return nil, fmt.Errorf("write request: %s", err)
	}

	// Read response from the server.
	resp, err := c.readResponse(conn)
	if err != nil {
		return nil, fmt.Errorf("read response: %s", err)
	}

	// If there was an error then return it and close connection.
	if resp.GetError() != "" {
		conn.Close()
		return nil, errors.New(resp.GetError())
	}

	// Returning remaining stream for caller to consume.
	return conn, nil
}

// writeRequest marshals and writes req to w.
func (c *Client) writeRequest(w io.Writer, req *internal.Request) error {
	// Marshal request.
	buf, err := proto.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %s", err)
	}

	// Write request length.
	if err := binary.Write(w, binary.BigEndian, uint32(len(buf))); err != nil {
		return fmt.Errorf("write request length: %s", err)
	}

	// Send request to server.
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("write request body: %s", err)
	}

	return nil
}

// readResponse reads and unmarshals a Response from r.
func (c *Client) readResponse(r io.Reader) (*internal.Response, error) {
	// Read response length.
	var n uint32
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, fmt.Errorf("read response length: %s", err)
	}

	// Read response.
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, fmt.Errorf("read response: %s", err)
	}

	// Unmarshal response.
	resp := &internal.Response{}
	if err := proto.Unmarshal(buf, resp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %s", err)
	}

	return resp, nil
}
