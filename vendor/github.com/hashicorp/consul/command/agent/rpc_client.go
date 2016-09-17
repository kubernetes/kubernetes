package agent

import (
	"bufio"
	"fmt"
	"github.com/hashicorp/go-msgpack/codec"
	"github.com/hashicorp/logutils"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	clientClosed = fmt.Errorf("client closed")
)

type seqCallback struct {
	handler func(*responseHeader)
}

func (sc *seqCallback) Handle(resp *responseHeader) {
	sc.handler(resp)
}
func (sc *seqCallback) Cleanup() {}

// seqHandler interface is used to handle responses
type seqHandler interface {
	Handle(*responseHeader)
	Cleanup()
}

// RPCClient is the RPC client to make requests to the agent RPC.
type RPCClient struct {
	seq uint64

	conn      net.Conn
	reader    *bufio.Reader
	writer    *bufio.Writer
	dec       *codec.Decoder
	enc       *codec.Encoder
	writeLock sync.Mutex

	dispatch     map[uint64]seqHandler
	dispatchLock sync.Mutex

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex
}

// send is used to send an object using the MsgPack encoding. send
// is serialized to prevent write overlaps, while properly buffering.
func (c *RPCClient) send(header *requestHeader, obj interface{}) error {
	c.writeLock.Lock()
	defer c.writeLock.Unlock()

	if c.shutdown {
		return clientClosed
	}

	if err := c.enc.Encode(header); err != nil {
		return err
	}

	if obj != nil {
		if err := c.enc.Encode(obj); err != nil {
			return err
		}
	}

	if err := c.writer.Flush(); err != nil {
		return err
	}

	return nil
}

// NewRPCClient is used to create a new RPC client given the address.
// This will properly dial, handshake, and start listening
func NewRPCClient(addr string) (*RPCClient, error) {
	var conn net.Conn
	var err error

	if envAddr := os.Getenv("CONSUL_RPC_ADDR"); envAddr != "" {
		addr = envAddr
	}

	// Try to dial to agent
	mode := "tcp"
	if strings.HasPrefix(addr, "/") {
		mode = "unix"
	}
	if conn, err = net.Dial(mode, addr); err != nil {
		return nil, err
	}

	// Create the client
	client := &RPCClient{
		seq:        0,
		conn:       conn,
		reader:     bufio.NewReader(conn),
		writer:     bufio.NewWriter(conn),
		dispatch:   make(map[uint64]seqHandler),
		shutdownCh: make(chan struct{}),
	}
	client.dec = codec.NewDecoder(client.reader, msgpackHandle)
	client.enc = codec.NewEncoder(client.writer, msgpackHandle)
	go client.listen()

	// Do the initial handshake
	if err := client.handshake(); err != nil {
		client.Close()
		return nil, err
	}
	return client, err
}

// StreamHandle is an opaque handle passed to stop to stop streaming
type StreamHandle uint64

// Close is used to free any resources associated with the client
func (c *RPCClient) Close() error {
	c.shutdownLock.Lock()
	defer c.shutdownLock.Unlock()

	if !c.shutdown {
		c.shutdown = true
		close(c.shutdownCh)
		c.deregisterAll()
		return c.conn.Close()
	}
	return nil
}

// ForceLeave is used to ask the agent to issue a leave command for
// a given node
func (c *RPCClient) ForceLeave(node string) error {
	header := requestHeader{
		Command: forceLeaveCommand,
		Seq:     c.getSeq(),
	}
	req := forceLeaveRequest{
		Node: node,
	}
	return c.genericRPC(&header, &req, nil)
}

// Join is used to instruct the agent to attempt a join
func (c *RPCClient) Join(addrs []string, wan bool) (int, error) {
	header := requestHeader{
		Command: joinCommand,
		Seq:     c.getSeq(),
	}
	req := joinRequest{
		Existing: addrs,
		WAN:      wan,
	}
	var resp joinResponse

	err := c.genericRPC(&header, &req, &resp)
	return int(resp.Num), err
}

// LANMembers is used to fetch a list of known members
func (c *RPCClient) LANMembers() ([]Member, error) {
	header := requestHeader{
		Command: membersLANCommand,
		Seq:     c.getSeq(),
	}
	var resp membersResponse

	err := c.genericRPC(&header, nil, &resp)
	return resp.Members, err
}

// WANMembers is used to fetch a list of known members
func (c *RPCClient) WANMembers() ([]Member, error) {
	header := requestHeader{
		Command: membersWANCommand,
		Seq:     c.getSeq(),
	}
	var resp membersResponse

	err := c.genericRPC(&header, nil, &resp)
	return resp.Members, err
}

func (c *RPCClient) ListKeys(token string) (keyringResponse, error) {
	header := requestHeader{
		Command: listKeysCommand,
		Seq:     c.getSeq(),
		Token:   token,
	}
	var resp keyringResponse
	err := c.genericRPC(&header, nil, &resp)
	return resp, err
}

func (c *RPCClient) InstallKey(key, token string) (keyringResponse, error) {
	header := requestHeader{
		Command: installKeyCommand,
		Seq:     c.getSeq(),
		Token:   token,
	}
	req := keyringRequest{key}
	var resp keyringResponse
	err := c.genericRPC(&header, &req, &resp)
	return resp, err
}

func (c *RPCClient) UseKey(key, token string) (keyringResponse, error) {
	header := requestHeader{
		Command: useKeyCommand,
		Seq:     c.getSeq(),
		Token:   token,
	}
	req := keyringRequest{key}
	var resp keyringResponse
	err := c.genericRPC(&header, &req, &resp)
	return resp, err
}

func (c *RPCClient) RemoveKey(key, token string) (keyringResponse, error) {
	header := requestHeader{
		Command: removeKeyCommand,
		Seq:     c.getSeq(),
		Token:   token,
	}
	req := keyringRequest{key}
	var resp keyringResponse
	err := c.genericRPC(&header, &req, &resp)
	return resp, err
}

// Leave is used to trigger a graceful leave and shutdown
func (c *RPCClient) Leave() error {
	header := requestHeader{
		Command: leaveCommand,
		Seq:     c.getSeq(),
	}
	return c.genericRPC(&header, nil, nil)
}

// Stats is used to get debugging state information
func (c *RPCClient) Stats() (map[string]map[string]string, error) {
	header := requestHeader{
		Command: statsCommand,
		Seq:     c.getSeq(),
	}
	var resp map[string]map[string]string

	err := c.genericRPC(&header, nil, &resp)
	return resp, err
}

// Reload is used to trigger a configuration reload
func (c *RPCClient) Reload() error {
	header := requestHeader{
		Command: reloadCommand,
		Seq:     c.getSeq(),
	}
	return c.genericRPC(&header, nil, nil)
}

type monitorHandler struct {
	client *RPCClient
	closed bool
	init   bool
	initCh chan<- error
	logCh  chan<- string
	seq    uint64
}

func (mh *monitorHandler) Handle(resp *responseHeader) {
	// Initialize on the first response
	if !mh.init {
		mh.init = true
		mh.initCh <- strToError(resp.Error)
		return
	}

	// Decode logs for all other responses
	var rec logRecord
	if err := mh.client.dec.Decode(&rec); err != nil {
		log.Printf("[ERR] Failed to decode log: %v", err)
		mh.client.deregisterHandler(mh.seq)
		return
	}
	select {
	case mh.logCh <- rec.Log:
	default:
		log.Printf("[ERR] Dropping log! Monitor channel full")
	}
}

func (mh *monitorHandler) Cleanup() {
	if !mh.closed {
		if !mh.init {
			mh.init = true
			mh.initCh <- fmt.Errorf("Stream closed")
		}
		close(mh.logCh)
		mh.closed = true
	}
}

// Monitor is used to subscribe to the logs of the agent
func (c *RPCClient) Monitor(level logutils.LogLevel, ch chan<- string) (StreamHandle, error) {
	// Setup the request
	seq := c.getSeq()
	header := requestHeader{
		Command: monitorCommand,
		Seq:     seq,
	}
	req := monitorRequest{
		LogLevel: string(level),
	}

	// Create a monitor handler
	initCh := make(chan error, 1)
	handler := &monitorHandler{
		client: c,
		initCh: initCh,
		logCh:  ch,
		seq:    seq,
	}
	c.handleSeq(seq, handler)

	// Send the request
	if err := c.send(&header, &req); err != nil {
		c.deregisterHandler(seq)
		return 0, err
	}

	// Wait for a response
	select {
	case err := <-initCh:
		return StreamHandle(seq), err
	case <-c.shutdownCh:
		c.deregisterHandler(seq)
		return 0, clientClosed
	}
}

// Stop is used to unsubscribe from logs or event streams
func (c *RPCClient) Stop(handle StreamHandle) error {
	// Deregister locally first to stop delivery
	c.deregisterHandler(uint64(handle))

	header := requestHeader{
		Command: stopCommand,
		Seq:     c.getSeq(),
	}
	req := stopRequest{
		Stop: uint64(handle),
	}
	return c.genericRPC(&header, &req, nil)
}

// handshake is used to perform the initial handshake on connect
func (c *RPCClient) handshake() error {
	header := requestHeader{
		Command: handshakeCommand,
		Seq:     c.getSeq(),
	}
	req := handshakeRequest{
		Version: MaxRPCVersion,
	}
	return c.genericRPC(&header, &req, nil)
}

// genericRPC is used to send a request and wait for an
// errorSequenceResponse, potentially returning an error
func (c *RPCClient) genericRPC(header *requestHeader, req interface{}, resp interface{}) error {
	// Setup a response handler
	errCh := make(chan error, 1)
	handler := func(respHeader *responseHeader) {
		if resp != nil {
			err := c.dec.Decode(resp)
			if err != nil {
				errCh <- err
				return
			}
		}
		errCh <- strToError(respHeader.Error)
	}
	c.handleSeq(header.Seq, &seqCallback{handler: handler})
	defer c.deregisterHandler(header.Seq)

	// Send the request
	if err := c.send(header, req); err != nil {
		return err
	}

	// Wait for a response
	select {
	case err := <-errCh:
		return err
	case <-c.shutdownCh:
		return clientClosed
	}
}

// strToError converts a string to an error if not blank
func strToError(s string) error {
	if s != "" {
		return fmt.Errorf(s)
	}
	return nil
}

// getSeq returns the next sequence number in a safe manner
func (c *RPCClient) getSeq() uint64 {
	return atomic.AddUint64(&c.seq, 1)
}

// deregisterAll is used to deregister all handlers
func (c *RPCClient) deregisterAll() {
	c.dispatchLock.Lock()
	defer c.dispatchLock.Unlock()

	for _, seqH := range c.dispatch {
		seqH.Cleanup()
	}
	c.dispatch = make(map[uint64]seqHandler)
}

// deregisterHandler is used to deregister a handler
func (c *RPCClient) deregisterHandler(seq uint64) {
	c.dispatchLock.Lock()
	seqH, ok := c.dispatch[seq]
	delete(c.dispatch, seq)
	c.dispatchLock.Unlock()

	if ok {
		seqH.Cleanup()
	}
}

// handleSeq is used to setup a handlerto wait on a response for
// a given sequence number.
func (c *RPCClient) handleSeq(seq uint64, handler seqHandler) {
	c.dispatchLock.Lock()
	defer c.dispatchLock.Unlock()
	c.dispatch[seq] = handler
}

// respondSeq is used to respond to a given sequence number
func (c *RPCClient) respondSeq(seq uint64, respHeader *responseHeader) {
	c.dispatchLock.Lock()
	seqL, ok := c.dispatch[seq]
	c.dispatchLock.Unlock()

	// Get a registered listener, ignore if none
	if ok {
		seqL.Handle(respHeader)
	}
}

// listen is used to processes data coming over the RPC channel,
// and wrote it to the correct destination based on seq no
func (c *RPCClient) listen() {
	defer c.Close()
	var respHeader responseHeader
	for {
		if err := c.dec.Decode(&respHeader); err != nil {
			if !c.shutdown {
				log.Printf("[ERR] agent.client: Failed to decode response header: %v", err)
			}
			break
		}
		c.respondSeq(respHeader.Seq, &respHeader)
	}
}
