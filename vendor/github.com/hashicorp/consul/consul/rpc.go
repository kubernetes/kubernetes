package consul

import (
	"crypto/tls"
	"fmt"
	"io"
	"math/rand"
	"net"
	"strings"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/state"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/yamux"
	"github.com/inconshreveable/muxado"
)

type RPCType byte

const (
	rpcConsul RPCType = iota
	rpcRaft
	rpcMultiplex
	rpcTLS
	rpcMultiplexV2
)

const (
	// maxQueryTime is used to bound the limit of a blocking query
	maxQueryTime = 600 * time.Second

	// defaultQueryTime is the amount of time we block waiting for a change
	// if no time is specified. Previously we would wait the maxQueryTime.
	defaultQueryTime = 300 * time.Second

	// jitterFraction is a the limit to the amount of jitter we apply
	// to a user specified MaxQueryTime. We divide the specified time by
	// the fraction. So 16 == 6.25% limit of jitter
	jitterFraction = 16

	// Warn if the Raft command is larger than this.
	// If it's over 1MB something is probably being abusive.
	raftWarnSize = 1024 * 1024

	// enqueueLimit caps how long we will wait to enqueue
	// a new Raft command. Something is probably wrong if this
	// value is ever reached. However, it prevents us from blocking
	// the requesting goroutine forever.
	enqueueLimit = 30 * time.Second
)

// listen is used to listen for incoming RPC connections
func (s *Server) listen() {
	for {
		// Accept a connection
		conn, err := s.rpcListener.Accept()
		if err != nil {
			if s.shutdown {
				return
			}
			s.logger.Printf("[ERR] consul.rpc: failed to accept RPC conn: %v", err)
			continue
		}

		go s.handleConn(conn, false)
		metrics.IncrCounter([]string{"consul", "rpc", "accept_conn"}, 1)
	}
}

// logConn is a wrapper around memberlist's LogConn so that we format references
// to "from" addresses in a consistent way. This is just a shorter name.
func logConn(conn net.Conn) string {
	return memberlist.LogConn(conn)
}

// handleConn is used to determine if this is a Raft or
// Consul type RPC connection and invoke the correct handler
func (s *Server) handleConn(conn net.Conn, isTLS bool) {
	// Read a single byte
	buf := make([]byte, 1)
	if _, err := conn.Read(buf); err != nil {
		if err != io.EOF {
			s.logger.Printf("[ERR] consul.rpc: failed to read byte: %v %s", err, logConn(conn))
		}
		conn.Close()
		return
	}

	// Enforce TLS if VerifyIncoming is set
	if s.config.VerifyIncoming && !isTLS && RPCType(buf[0]) != rpcTLS {
		s.logger.Printf("[WARN] consul.rpc: Non-TLS connection attempted with VerifyIncoming set %s", logConn(conn))
		conn.Close()
		return
	}

	// Switch on the byte
	switch RPCType(buf[0]) {
	case rpcConsul:
		s.handleConsulConn(conn)

	case rpcRaft:
		metrics.IncrCounter([]string{"consul", "rpc", "raft_handoff"}, 1)
		s.raftLayer.Handoff(conn)

	case rpcMultiplex:
		s.handleMultiplex(conn)

	case rpcTLS:
		if s.rpcTLS == nil {
			s.logger.Printf("[WARN] consul.rpc: TLS connection attempted, server not configured for TLS %s", logConn(conn))
			conn.Close()
			return
		}
		conn = tls.Server(conn, s.rpcTLS)
		s.handleConn(conn, true)

	case rpcMultiplexV2:
		s.handleMultiplexV2(conn)

	default:
		s.logger.Printf("[ERR] consul.rpc: unrecognized RPC byte: %v %s", buf[0], logConn(conn))
		conn.Close()
		return
	}
}

// handleMultiplex is used to multiplex a single incoming connection
// using the Muxado multiplexer
func (s *Server) handleMultiplex(conn net.Conn) {
	defer conn.Close()
	server := muxado.Server(conn)
	for {
		sub, err := server.Accept()
		if err != nil {
			if !strings.Contains(err.Error(), "closed") {
				s.logger.Printf("[ERR] consul.rpc: multiplex conn accept failed: %v %s", err, logConn(conn))
			}
			return
		}
		go s.handleConsulConn(sub)
	}
}

// handleMultiplexV2 is used to multiplex a single incoming connection
// using the Yamux multiplexer
func (s *Server) handleMultiplexV2(conn net.Conn) {
	defer conn.Close()
	conf := yamux.DefaultConfig()
	conf.LogOutput = s.config.LogOutput
	server, _ := yamux.Server(conn, conf)
	for {
		sub, err := server.Accept()
		if err != nil {
			if err != io.EOF {
				s.logger.Printf("[ERR] consul.rpc: multiplex conn accept failed: %v %s", err, logConn(conn))
			}
			return
		}
		go s.handleConsulConn(sub)
	}
}

// handleConsulConn is used to service a single Consul RPC connection
func (s *Server) handleConsulConn(conn net.Conn) {
	defer conn.Close()
	rpcCodec := msgpackrpc.NewServerCodec(conn)
	for {
		select {
		case <-s.shutdownCh:
			return
		default:
		}

		if err := s.rpcServer.ServeRequest(rpcCodec); err != nil {
			if err != io.EOF && !strings.Contains(err.Error(), "closed") {
				s.logger.Printf("[ERR] consul.rpc: RPC error: %v %s", err, logConn(conn))
				metrics.IncrCounter([]string{"consul", "rpc", "request_error"}, 1)
			}
			return
		}
		metrics.IncrCounter([]string{"consul", "rpc", "request"}, 1)
	}
}

// forward is used to forward to a remote DC or to forward to the local leader
// Returns a bool of if forwarding was performed, as well as any error
func (s *Server) forward(method string, info structs.RPCInfo, args interface{}, reply interface{}) (bool, error) {
	// Handle DC forwarding
	dc := info.RequestDatacenter()
	if dc != s.config.Datacenter {
		err := s.forwardDC(method, dc, args, reply)
		return true, err
	}

	// Check if we can allow a stale read
	if info.IsRead() && info.AllowStaleRead() {
		return false, nil
	}

	// Handle leader forwarding
	if !s.IsLeader() {
		err := s.forwardLeader(method, args, reply)
		return true, err
	}
	return false, nil
}

// forwardLeader is used to forward an RPC call to the leader, or fail if no leader
func (s *Server) forwardLeader(method string, args interface{}, reply interface{}) error {
	// Get the leader
	leader := s.raft.Leader()
	if leader == "" {
		return structs.ErrNoLeader
	}

	// Lookup the server
	s.localLock.RLock()
	server := s.localConsuls[leader]
	s.localLock.RUnlock()

	// Handle a missing server
	if server == nil {
		return structs.ErrNoLeader
	}
	return s.connPool.RPC(s.config.Datacenter, server.Addr, server.Version, method, args, reply)
}

// forwardDC is used to forward an RPC call to a remote DC, or fail if no servers
func (s *Server) forwardDC(method, dc string, args interface{}, reply interface{}) error {
	// Bail if we can't find any servers
	s.remoteLock.RLock()
	servers := s.remoteConsuls[dc]
	if len(servers) == 0 {
		s.remoteLock.RUnlock()
		s.logger.Printf("[WARN] consul.rpc: RPC request for DC '%s', no path found", dc)
		return structs.ErrNoDCPath
	}

	// Select a random addr
	offset := rand.Int31() % int32(len(servers))
	server := servers[offset]
	s.remoteLock.RUnlock()

	// Forward to remote Consul
	metrics.IncrCounter([]string{"consul", "rpc", "cross-dc", dc}, 1)
	return s.connPool.RPC(dc, server.Addr, server.Version, method, args, reply)
}

// globalRPC is used to forward an RPC request to one server in each datacenter.
// This will only error for RPC-related errors. Otherwise, application-level
// errors can be sent in the response objects.
func (s *Server) globalRPC(method string, args interface{},
	reply structs.CompoundResponse) error {

	errorCh := make(chan error)
	respCh := make(chan interface{})

	// Make a new request into each datacenter
	for dc, _ := range s.remoteConsuls {
		go func(dc string) {
			rr := reply.New()
			if err := s.forwardDC(method, dc, args, &rr); err != nil {
				errorCh <- err
				return
			}
			respCh <- rr
		}(dc)
	}

	replies, total := 0, len(s.remoteConsuls)
	for replies < total {
		select {
		case err := <-errorCh:
			return err
		case rr := <-respCh:
			reply.Add(rr)
			replies++
		}
	}
	return nil
}

// raftApply is used to encode a message, run it through raft, and return
// the FSM response along with any errors
func (s *Server) raftApply(t structs.MessageType, msg interface{}) (interface{}, error) {
	buf, err := structs.Encode(t, msg)
	if err != nil {
		return nil, fmt.Errorf("Failed to encode request: %v", err)
	}

	// Warn if the command is very large
	if n := len(buf); n > raftWarnSize {
		s.logger.Printf("[WARN] consul: Attempting to apply large raft entry (%d bytes)", n)
	}

	future := s.raft.Apply(buf, enqueueLimit)
	if err := future.Error(); err != nil {
		return nil, err
	}

	return future.Response(), nil
}

// blockingRPC is used for queries that need to wait for a minimum index. This
// is used to block and wait for changes.
func (s *Server) blockingRPC(queryOpts *structs.QueryOptions, queryMeta *structs.QueryMeta,
	watch state.Watch, run func() error) error {
	var timeout *time.Timer
	var notifyCh chan struct{}

	// Fast path right to the non-blocking query.
	if queryOpts.MinQueryIndex == 0 {
		goto RUN_QUERY
	}

	// Make sure a watch was given if we were asked to block.
	if watch == nil {
		panic("no watch given for blocking query")
	}

	// Restrict the max query time, and ensure there is always one.
	if queryOpts.MaxQueryTime > maxQueryTime {
		queryOpts.MaxQueryTime = maxQueryTime
	} else if queryOpts.MaxQueryTime <= 0 {
		queryOpts.MaxQueryTime = defaultQueryTime
	}

	// Apply a small amount of jitter to the request.
	queryOpts.MaxQueryTime += lib.RandomStagger(queryOpts.MaxQueryTime / jitterFraction)

	// Setup a query timeout.
	timeout = time.NewTimer(queryOpts.MaxQueryTime)

	// Setup the notify channel.
	notifyCh = make(chan struct{}, 1)

	// Ensure we tear down any watches on return.
	defer func() {
		timeout.Stop()
		watch.Clear(notifyCh)
	}()

REGISTER_NOTIFY:
	// Register the notification channel. This may be done multiple times if
	// we haven't reached the target wait index.
	watch.Wait(notifyCh)

RUN_QUERY:
	// Update the query metadata.
	s.setQueryMeta(queryMeta)

	// If the read must be consistent we verify that we are still the leader.
	if queryOpts.RequireConsistent {
		if err := s.consistentRead(); err != nil {
			return err
		}
	}

	// Run the query.
	metrics.IncrCounter([]string{"consul", "rpc", "query"}, 1)
	err := run()

	// Check for minimum query time.
	if err == nil && queryMeta.Index > 0 && queryMeta.Index <= queryOpts.MinQueryIndex {
		select {
		case <-notifyCh:
			goto REGISTER_NOTIFY
		case <-timeout.C:
		}
	}
	return err
}

// setQueryMeta is used to populate the QueryMeta data for an RPC call
func (s *Server) setQueryMeta(m *structs.QueryMeta) {
	if s.IsLeader() {
		m.LastContact = 0
		m.KnownLeader = true
	} else {
		m.LastContact = time.Now().Sub(s.raft.LastContact())
		m.KnownLeader = (s.raft.Leader() != "")
	}
}

// consistentRead is used to ensure we do not perform a stale
// read. This is done by verifying leadership before the read.
func (s *Server) consistentRead() error {
	defer metrics.MeasureSince([]string{"consul", "rpc", "consistentRead"}, time.Now())
	future := s.raft.VerifyLeader()
	return future.Error()
}
