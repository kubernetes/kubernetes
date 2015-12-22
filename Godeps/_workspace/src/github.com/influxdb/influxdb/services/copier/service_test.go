package copier_test

import (
	"bytes"
	"encoding/binary"
	"io"
	"io/ioutil"
	"log"
	"net"
	"os"
	"path/filepath"
	"testing"

	"github.com/influxdb/influxdb/services/copier"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/tsdb"
	_ "github.com/influxdb/influxdb/tsdb/engine"
)

// Ensure the service can return shard data.
func TestService_handleConn(t *testing.T) {
	t.Skip("not implemented for tsm1 engine")
	s := MustOpenService()
	defer s.Close()

	// Mock shard.
	sh := MustOpenShard(123)
	defer sh.Close()
	s.TSDBStore.ShardFn = func(id uint64) *tsdb.Shard {
		if id != 123 {
			t.Fatalf("unexpected id: %d", id)
		}
		return sh.Shard
	}

	// Create client and request shard from service.
	c := copier.NewClient(s.Addr().String())
	r, err := c.ShardReader(123)
	if err != nil {
		t.Fatal(err)
	} else if r == nil {
		t.Fatal("expected reader")
	}
	defer r.Close()

	// Slurp from reader.
	var n uint64
	if err := binary.Read(r, binary.BigEndian, &n); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		t.Fatal(err)
	}

	// Read database from disk.
	exp, err := ioutil.ReadFile(sh.Path())
	if err != nil {
		t.Fatal(err)
	}

	// Trim expected bytes since bolt won't read beyond the HWM.
	exp = exp[0:len(buf)]

	// Compare disk and reader contents.
	if !bytes.Equal(exp, buf) {
		t.Fatalf("data mismatch: exp=len(%d), got=len(%d)", len(exp), len(buf))
	}
}

// Ensure the service can return an error to the client.
func TestService_handleConn_Error(t *testing.T) {
	s := MustOpenService()
	defer s.Close()

	// Mock missing shard.
	s.TSDBStore.ShardFn = func(id uint64) *tsdb.Shard { return nil }

	// Create client and request shard from service.
	c := copier.NewClient(s.Addr().String())
	r, err := c.ShardReader(123)
	if err == nil || err.Error() != `shard not found: id=123` {
		t.Fatalf("unexpected error: %s", err)
	} else if r != nil {
		t.Fatal("expected nil reader")
	}
}

// Service represents a test wrapper for copier.Service.
type Service struct {
	*copier.Service

	ln        net.Listener
	TSDBStore ServiceTSDBStore
}

// NewService returns a new instance of Service.
func NewService() *Service {
	s := &Service{
		Service: copier.NewService(),
	}
	s.Service.TSDBStore = &s.TSDBStore

	if !testing.Verbose() {
		s.SetLogger(log.New(ioutil.Discard, "", 0))
	}
	return s
}

// MustOpenService returns a new, opened service. Panic on error.
func MustOpenService() *Service {
	// Open randomly assigned port.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}

	// Start muxer.
	mux := tcp.NewMux()

	// Create new service and attach mux'd listener.
	s := NewService()
	s.ln = ln
	s.Listener = mux.Listen(copier.MuxHeader)
	go mux.Serve(ln)

	if err := s.Open(); err != nil {
		panic(err)
	}

	return s
}

// Close shuts down the service and the attached listener.
func (s *Service) Close() error {
	s.ln.Close()
	err := s.Service.Close()
	return err
}

// Addr returns the address of the service.
func (s *Service) Addr() net.Addr { return s.ln.Addr() }

// ServiceTSDBStore is a mock that implements copier.Service.TSDBStore.
type ServiceTSDBStore struct {
	ShardFn func(id uint64) *tsdb.Shard
}

func (ss *ServiceTSDBStore) Shard(id uint64) *tsdb.Shard { return ss.ShardFn(id) }

// Shard is a test wrapper for tsdb.Shard.
type Shard struct {
	*tsdb.Shard
	path string
}

// MustOpenShard returns a temporary, opened shard.
func MustOpenShard(id uint64) *Shard {
	path, err := ioutil.TempDir("", "copier-")
	if err != nil {
		panic(err)
	}

	sh := &Shard{
		Shard: tsdb.NewShard(id,
			tsdb.NewDatabaseIndex(),
			filepath.Join(path, "data"),
			filepath.Join(path, "wal"),
			tsdb.NewEngineOptions(),
		),
		path: path,
	}
	if err := sh.Open(); err != nil {
		sh.Close()
		panic(err)
	}

	return sh
}

func (sh *Shard) Close() error {
	err := sh.Shard.Close()
	os.RemoveAll(sh.Path())
	return err
}
