package agent

/*
 The agent exposes an RPC mechanism that is used for both controlling
 Consul as well as providing a fast streaming mechanism for events. This
 allows other applications to easily leverage Consul without embedding.

 We additionally make use of the RPC layer to also handle calls from
 the CLI to unify the code paths. This results in a split Request/Response
 as well as streaming mode of operation.

 The system is fairly simple, each client opens a TCP connection to the
 agent. The connection is initialized with a handshake which establishes
 the protocol version being used. This is to allow for future changes to
 the protocol.

 Once initialized, clients send commands and wait for responses. Certain
 commands will cause the client to subscribe to events, and those will be
 pushed down the socket as they are received. This provides a low-latency
 mechanism for applications to send and receive events, while also providing
 a flexible control mechanism for Consul.
*/

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/go-msgpack/codec"
	"github.com/hashicorp/logutils"
	"github.com/hashicorp/serf/serf"
)

const (
	MinRPCVersion = 1
	MaxRPCVersion = 1
)

const (
	handshakeCommand  = "handshake"
	forceLeaveCommand = "force-leave"
	joinCommand       = "join"
	membersLANCommand = "members-lan"
	membersWANCommand = "members-wan"
	stopCommand       = "stop"
	monitorCommand    = "monitor"
	leaveCommand      = "leave"
	statsCommand      = "stats"
	reloadCommand     = "reload"
	installKeyCommand = "install-key"
	useKeyCommand     = "use-key"
	removeKeyCommand  = "remove-key"
	listKeysCommand   = "list-keys"
)

const (
	unsupportedCommand    = "Unsupported command"
	unsupportedRPCVersion = "Unsupported RPC version"
	duplicateHandshake    = "Handshake already performed"
	handshakeRequired     = "Handshake required"
	monitorExists         = "Monitor already exists"
)

// msgpackHandle is a shared handle for encoding/decoding of
// messages
var msgpackHandle = &codec.MsgpackHandle{
	RawToString: true,
	WriteExt:    true,
}

// Request header is sent before each request
type requestHeader struct {
	Command string
	Seq     uint64
	Token   string
}

// Response header is sent before each response
type responseHeader struct {
	Seq   uint64
	Error string
}

type handshakeRequest struct {
	Version int32
}

type forceLeaveRequest struct {
	Node string
}

type joinRequest struct {
	Existing []string
	WAN      bool
}

type joinResponse struct {
	Num int32
}

type keyringRequest struct {
	Key string
}

type KeyringEntry struct {
	Datacenter string
	Pool       string
	Key        string
	Count      int
}

type KeyringMessage struct {
	Datacenter string
	Pool       string
	Node       string
	Message    string
}

type KeyringInfo struct {
	Datacenter string
	Pool       string
	NumNodes   int
	Error      string
}

type keyringResponse struct {
	Keys     []KeyringEntry
	Messages []KeyringMessage
	Info     []KeyringInfo
}

type membersResponse struct {
	Members []Member
}

type monitorRequest struct {
	LogLevel string
}

type stopRequest struct {
	Stop uint64
}

type logRecord struct {
	Log string
}

type Member struct {
	Name        string
	Addr        net.IP
	Tags        map[string]string
	Status      string
	Port        uint16
	ProtocolMin uint8
	ProtocolMax uint8
	ProtocolCur uint8
	DelegateMin uint8
	DelegateMax uint8
	DelegateCur uint8
}

type AgentRPC struct {
	sync.Mutex
	agent     *Agent
	clients   map[string]*rpcClient
	listener  net.Listener
	logger    *log.Logger
	logWriter *logWriter
	reloadCh  chan struct{}
	stop      bool
	stopCh    chan struct{}
}

type rpcClient struct {
	name        string
	conn        net.Conn
	reader      *bufio.Reader
	writer      *bufio.Writer
	dec         *codec.Decoder
	enc         *codec.Encoder
	writeLock   sync.Mutex
	version     int32 // From the handshake, 0 before
	logStreamer *logStream
}

// send is used to send an object using the MsgPack encoding. send
// is serialized to prevent write overlaps, while properly buffering.
func (c *rpcClient) Send(header *responseHeader, obj interface{}) error {
	c.writeLock.Lock()
	defer c.writeLock.Unlock()

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

func (c *rpcClient) String() string {
	return fmt.Sprintf("rpc.client: %v", c.conn)
}

// NewAgentRPC is used to create a new Agent RPC handler
func NewAgentRPC(agent *Agent, listener net.Listener,
	logOutput io.Writer, logWriter *logWriter) *AgentRPC {
	if logOutput == nil {
		logOutput = os.Stderr
	}
	rpc := &AgentRPC{
		agent:     agent,
		clients:   make(map[string]*rpcClient),
		listener:  listener,
		logger:    log.New(logOutput, "", log.LstdFlags),
		logWriter: logWriter,
		reloadCh:  make(chan struct{}, 1),
		stopCh:    make(chan struct{}),
	}
	go rpc.listen()
	return rpc
}

// Shutdown is used to shutdown the RPC layer
func (i *AgentRPC) Shutdown() {
	i.Lock()
	defer i.Unlock()

	if i.stop {
		return
	}

	i.stop = true
	close(i.stopCh)
	i.listener.Close()

	// Close the existing connections
	for _, client := range i.clients {
		client.conn.Close()
	}
}

// ReloadCh returns a channel that can be watched for
// when a reload is being triggered.
func (i *AgentRPC) ReloadCh() <-chan struct{} {
	return i.reloadCh
}

// listen is a long running routine that listens for new clients
func (i *AgentRPC) listen() {
	for {
		conn, err := i.listener.Accept()
		if err != nil {
			if i.stop {
				return
			}
			i.logger.Printf("[ERR] agent.rpc: Failed to accept client: %v", err)
			continue
		}
		i.logger.Printf("[INFO] agent.rpc: Accepted client: %v", conn.RemoteAddr())

		// Wrap the connection in a client
		client := &rpcClient{
			name:   conn.RemoteAddr().String(),
			conn:   conn,
			reader: bufio.NewReader(conn),
			writer: bufio.NewWriter(conn),
		}
		client.dec = codec.NewDecoder(client.reader, msgpackHandle)
		client.enc = codec.NewEncoder(client.writer, msgpackHandle)

		// Register the client
		i.Lock()
		if !i.stop {
			i.clients[client.name] = client
			go i.handleClient(client)
		} else {
			conn.Close()
		}
		i.Unlock()
	}
}

// deregisterClient is called to cleanup after a client disconnects
func (i *AgentRPC) deregisterClient(client *rpcClient) {
	// Close the socket
	client.conn.Close()

	// Remove from the clients list
	i.Lock()
	delete(i.clients, client.name)
	i.Unlock()

	// Remove from the log writer
	if client.logStreamer != nil {
		i.logWriter.DeregisterHandler(client.logStreamer)
		client.logStreamer.Stop()
	}
}

// handleClient is a long running routine that handles a single client
func (i *AgentRPC) handleClient(client *rpcClient) {
	defer i.deregisterClient(client)
	var reqHeader requestHeader
	for {
		// Decode the header
		if err := client.dec.Decode(&reqHeader); err != nil {
			if !i.stop {
				// The second part of this if is to block socket
				// errors from Windows which appear to happen every
				// time there is an EOF.
				if err != io.EOF && !strings.Contains(err.Error(), "WSARecv") {
					i.logger.Printf("[ERR] agent.rpc: failed to decode request header: %v", err)
				}
			}
			return
		}

		// Evaluate the command
		if err := i.handleRequest(client, &reqHeader); err != nil {
			i.logger.Printf("[ERR] agent.rpc: Failed to evaluate request: %v", err)
			return
		}
	}
}

// handleRequest is used to evaluate a single client command
func (i *AgentRPC) handleRequest(client *rpcClient, reqHeader *requestHeader) error {
	// Look for a command field
	command := reqHeader.Command
	seq := reqHeader.Seq
	token := reqHeader.Token

	// Ensure the handshake is performed before other commands
	if command != handshakeCommand && client.version == 0 {
		respHeader := responseHeader{Seq: seq, Error: handshakeRequired}
		client.Send(&respHeader, nil)
		return fmt.Errorf(handshakeRequired)
	}

	// Dispatch command specific handlers
	switch command {
	case handshakeCommand:
		return i.handleHandshake(client, seq)

	case membersLANCommand:
		return i.handleMembersLAN(client, seq)

	case membersWANCommand:
		return i.handleMembersWAN(client, seq)

	case monitorCommand:
		return i.handleMonitor(client, seq)

	case stopCommand:
		return i.handleStop(client, seq)

	case forceLeaveCommand:
		return i.handleForceLeave(client, seq)

	case joinCommand:
		return i.handleJoin(client, seq)

	case leaveCommand:
		return i.handleLeave(client, seq)

	case statsCommand:
		return i.handleStats(client, seq)

	case reloadCommand:
		return i.handleReload(client, seq)

	case installKeyCommand, useKeyCommand, removeKeyCommand, listKeysCommand:
		return i.handleKeyring(client, seq, command, token)

	default:
		respHeader := responseHeader{Seq: seq, Error: unsupportedCommand}
		client.Send(&respHeader, nil)
		return fmt.Errorf("command '%s' not recognized", command)
	}
}

func (i *AgentRPC) handleHandshake(client *rpcClient, seq uint64) error {
	var req handshakeRequest
	if err := client.dec.Decode(&req); err != nil {
		return fmt.Errorf("decode failed: %v", err)
	}

	resp := responseHeader{
		Seq:   seq,
		Error: "",
	}

	// Check the version
	if req.Version < MinRPCVersion || req.Version > MaxRPCVersion {
		resp.Error = unsupportedRPCVersion
	} else if client.version != 0 {
		resp.Error = duplicateHandshake
	} else {
		client.version = req.Version
	}
	return client.Send(&resp, nil)
}

func (i *AgentRPC) handleForceLeave(client *rpcClient, seq uint64) error {
	var req forceLeaveRequest
	if err := client.dec.Decode(&req); err != nil {
		return fmt.Errorf("decode failed: %v", err)
	}

	// Attempt leave
	err := i.agent.ForceLeave(req.Node)

	// Respond
	resp := responseHeader{
		Seq:   seq,
		Error: errToString(err),
	}
	return client.Send(&resp, nil)
}

func (i *AgentRPC) handleJoin(client *rpcClient, seq uint64) error {
	var req joinRequest
	if err := client.dec.Decode(&req); err != nil {
		return fmt.Errorf("decode failed: %v", err)
	}

	// Attempt the join
	var num int
	var err error
	if req.WAN {
		num, err = i.agent.JoinWAN(req.Existing)
	} else {
		num, err = i.agent.JoinLAN(req.Existing)
	}

	// Respond
	header := responseHeader{
		Seq:   seq,
		Error: errToString(err),
	}
	resp := joinResponse{
		Num: int32(num),
	}
	return client.Send(&header, &resp)
}

func (i *AgentRPC) handleMembersLAN(client *rpcClient, seq uint64) error {
	raw := i.agent.LANMembers()
	return formatMembers(raw, client, seq)
}

func (i *AgentRPC) handleMembersWAN(client *rpcClient, seq uint64) error {
	raw := i.agent.WANMembers()
	return formatMembers(raw, client, seq)
}

func formatMembers(raw []serf.Member, client *rpcClient, seq uint64) error {
	members := make([]Member, 0, len(raw))
	for _, m := range raw {
		sm := Member{
			Name:        m.Name,
			Addr:        m.Addr,
			Port:        m.Port,
			Tags:        m.Tags,
			Status:      m.Status.String(),
			ProtocolMin: m.ProtocolMin,
			ProtocolMax: m.ProtocolMax,
			ProtocolCur: m.ProtocolCur,
			DelegateMin: m.DelegateMin,
			DelegateMax: m.DelegateMax,
			DelegateCur: m.DelegateCur,
		}
		members = append(members, sm)
	}

	header := responseHeader{
		Seq:   seq,
		Error: "",
	}
	resp := membersResponse{
		Members: members,
	}
	return client.Send(&header, &resp)
}

func (i *AgentRPC) handleMonitor(client *rpcClient, seq uint64) error {
	var req monitorRequest
	if err := client.dec.Decode(&req); err != nil {
		return fmt.Errorf("decode failed: %v", err)
	}

	resp := responseHeader{
		Seq:   seq,
		Error: "",
	}

	// Upper case the log level
	req.LogLevel = strings.ToUpper(req.LogLevel)

	// Create a level filter
	filter := LevelFilter()
	filter.MinLevel = logutils.LogLevel(req.LogLevel)
	if !ValidateLevelFilter(filter.MinLevel, filter) {
		resp.Error = fmt.Sprintf("Unknown log level: %s", filter.MinLevel)
		goto SEND
	}

	// Check if there is an existing monitor
	if client.logStreamer != nil {
		resp.Error = monitorExists
		goto SEND
	}

	// Create a log streamer
	client.logStreamer = newLogStream(client, filter, seq, i.logger)

	// Register with the log writer. Defer so that we can respond before
	// registration, avoids any possible race condition
	defer i.logWriter.RegisterHandler(client.logStreamer)

SEND:
	return client.Send(&resp, nil)
}

func (i *AgentRPC) handleStop(client *rpcClient, seq uint64) error {
	var req stopRequest
	if err := client.dec.Decode(&req); err != nil {
		return fmt.Errorf("decode failed: %v", err)
	}

	// Remove a log monitor if any
	if client.logStreamer != nil && client.logStreamer.seq == req.Stop {
		i.logWriter.DeregisterHandler(client.logStreamer)
		client.logStreamer.Stop()
		client.logStreamer = nil
	}

	// Always succeed
	resp := responseHeader{Seq: seq, Error: ""}
	return client.Send(&resp, nil)
}

func (i *AgentRPC) handleLeave(client *rpcClient, seq uint64) error {
	i.logger.Printf("[INFO] agent.rpc: Graceful leave triggered")

	// Do the leave
	err := i.agent.Leave()
	if err != nil {
		i.logger.Printf("[ERR] agent.rpc: leave failed: %v", err)
	}
	resp := responseHeader{Seq: seq, Error: errToString(err)}

	// Send and wait
	err = client.Send(&resp, nil)

	// Trigger a shutdown!
	if err := i.agent.Shutdown(); err != nil {
		i.logger.Printf("[ERR] agent.rpc: shutdown failed: %v", err)
	}
	return err
}

// handleStats is used to get various statistics
func (i *AgentRPC) handleStats(client *rpcClient, seq uint64) error {
	header := responseHeader{
		Seq:   seq,
		Error: "",
	}
	resp := i.agent.Stats()
	return client.Send(&header, resp)
}

func (i *AgentRPC) handleReload(client *rpcClient, seq uint64) error {
	// Push to the reload channel
	select {
	case i.reloadCh <- struct{}{}:
	default:
	}

	// Always succeed
	resp := responseHeader{Seq: seq, Error: ""}
	return client.Send(&resp, nil)
}

func (i *AgentRPC) handleKeyring(client *rpcClient, seq uint64, cmd, token string) error {
	var req keyringRequest
	var queryResp *structs.KeyringResponses
	var r keyringResponse
	var err error

	if cmd != listKeysCommand {
		if err = client.dec.Decode(&req); err != nil {
			return fmt.Errorf("decode failed: %v", err)
		}
	}

	switch cmd {
	case listKeysCommand:
		queryResp, err = i.agent.ListKeys(token)
	case installKeyCommand:
		queryResp, err = i.agent.InstallKey(req.Key, token)
	case useKeyCommand:
		queryResp, err = i.agent.UseKey(req.Key, token)
	case removeKeyCommand:
		queryResp, err = i.agent.RemoveKey(req.Key, token)
	default:
		respHeader := responseHeader{Seq: seq, Error: unsupportedCommand}
		client.Send(&respHeader, nil)
		return fmt.Errorf("command '%s' not recognized", cmd)
	}

	header := responseHeader{
		Seq:   seq,
		Error: errToString(err),
	}

	if queryResp == nil {
		goto SEND
	}

	for _, kr := range queryResp.Responses {
		var pool string
		if kr.WAN {
			pool = "WAN"
		} else {
			pool = "LAN"
		}
		for node, message := range kr.Messages {
			msg := KeyringMessage{
				Datacenter: kr.Datacenter,
				Pool:       pool,
				Node:       node,
				Message:    message,
			}
			r.Messages = append(r.Messages, msg)
		}
		for key, qty := range kr.Keys {
			k := KeyringEntry{
				Datacenter: kr.Datacenter,
				Pool:       pool,
				Key:        key,
				Count:      qty,
			}
			r.Keys = append(r.Keys, k)
		}
		info := KeyringInfo{
			Datacenter: kr.Datacenter,
			Pool:       pool,
			NumNodes:   kr.NumNodes,
			Error:      kr.Error,
		}
		r.Info = append(r.Info, info)
	}

SEND:
	return client.Send(&header, r)
}

// Used to convert an error to a string representation
func errToString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
