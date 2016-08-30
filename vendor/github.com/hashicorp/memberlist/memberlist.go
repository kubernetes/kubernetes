/*
memberlist is a library that manages cluster
membership and member failure detection using a gossip based protocol.

The use cases for such a library are far-reaching: all distributed systems
require membership, and memberlist is a re-usable solution to managing
cluster membership and node failure detection.

memberlist is eventually consistent but converges quickly on average.
The speed at which it converges can be heavily tuned via various knobs
on the protocol. Node failures are detected and network partitions are partially
tolerated by attempting to communicate to potentially dead nodes through
multiple routes.
*/
package memberlist

import (
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"sync"
	"time"
)

type Memberlist struct {
	sequenceNum uint32 // Local sequence number
	incarnation uint32 // Local incarnation number
	numNodes    uint32 // Number of known nodes (estimate)

	config         *Config
	shutdown       bool
	shutdownCh     chan struct{}
	leave          bool
	leaveBroadcast chan struct{}

	udpListener *net.UDPConn
	tcpListener *net.TCPListener
	handoff     chan msgHandoff

	nodeLock sync.RWMutex
	nodes    []*nodeState          // Known nodes
	nodeMap  map[string]*nodeState // Maps Addr.String() -> NodeState

	tickerLock sync.Mutex
	tickers    []*time.Ticker
	stopTick   chan struct{}
	probeIndex int

	ackLock     sync.Mutex
	ackHandlers map[uint32]*ackHandler

	broadcasts *TransmitLimitedQueue

	logger *log.Logger
}

// newMemberlist creates the network listeners.
// Does not schedule execution of background maintenence.
func newMemberlist(conf *Config) (*Memberlist, error) {
	if conf.ProtocolVersion < ProtocolVersionMin {
		return nil, fmt.Errorf("Protocol version '%d' too low. Must be in range: [%d, %d]",
			conf.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	} else if conf.ProtocolVersion > ProtocolVersionMax {
		return nil, fmt.Errorf("Protocol version '%d' too high. Must be in range: [%d, %d]",
			conf.ProtocolVersion, ProtocolVersionMin, ProtocolVersionMax)
	}

	if len(conf.SecretKey) > 0 {
		if conf.Keyring == nil {
			keyring, err := NewKeyring(nil, conf.SecretKey)
			if err != nil {
				return nil, err
			}
			conf.Keyring = keyring
		} else {
			if err := conf.Keyring.AddKey(conf.SecretKey); err != nil {
				return nil, err
			}
			if err := conf.Keyring.UseKey(conf.SecretKey); err != nil {
				return nil, err
			}
		}
	}

	tcpAddr := &net.TCPAddr{IP: net.ParseIP(conf.BindAddr), Port: conf.BindPort}
	tcpLn, err := net.ListenTCP("tcp", tcpAddr)
	if err != nil {
		return nil, fmt.Errorf("Failed to start TCP listener. Err: %s", err)
	}
	if conf.BindPort == 0 {
		conf.BindPort = tcpLn.Addr().(*net.TCPAddr).Port
	}

	udpAddr := &net.UDPAddr{IP: net.ParseIP(conf.BindAddr), Port: conf.BindPort}
	udpLn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		tcpLn.Close()
		return nil, fmt.Errorf("Failed to start UDP listener. Err: %s", err)
	}

	// Set the UDP receive window size
	setUDPRecvBuf(udpLn)

	if conf.LogOutput != nil && conf.Logger != nil {
		return nil, fmt.Errorf("Cannot specify both LogOutput and Logger. Please choose a single log configuration setting.")
	}

	logDest := conf.LogOutput
	if logDest == nil {
		logDest = os.Stderr
	}

	logger := conf.Logger
	if logger == nil {
		logger = log.New(logDest, "", log.LstdFlags)
	}

	m := &Memberlist{
		config:         conf,
		shutdownCh:     make(chan struct{}),
		leaveBroadcast: make(chan struct{}, 1),
		udpListener:    udpLn,
		tcpListener:    tcpLn,
		handoff:        make(chan msgHandoff, 1024),
		nodeMap:        make(map[string]*nodeState),
		ackHandlers:    make(map[uint32]*ackHandler),
		broadcasts:     &TransmitLimitedQueue{RetransmitMult: conf.RetransmitMult},
		logger:         logger,
	}
	m.broadcasts.NumNodes = func() int {
		return m.estNumNodes()
	}
	go m.tcpListen()
	go m.udpListen()
	go m.udpHandler()
	return m, nil
}

// Create will create a new Memberlist using the given configuration.
// This will not connect to any other node (see Join) yet, but will start
// all the listeners to allow other nodes to join this memberlist.
// After creating a Memberlist, the configuration given should not be
// modified by the user anymore.
func Create(conf *Config) (*Memberlist, error) {
	m, err := newMemberlist(conf)
	if err != nil {
		return nil, err
	}
	if err := m.setAlive(); err != nil {
		m.Shutdown()
		return nil, err
	}
	m.schedule()
	return m, nil
}

// Join is used to take an existing Memberlist and attempt to join a cluster
// by contacting all the given hosts and performing a state sync. Initially,
// the Memberlist only contains our own state, so doing this will cause
// remote nodes to become aware of the existence of this node, effectively
// joining the cluster.
//
// This returns the number of hosts successfully contacted and an error if
// none could be reached. If an error is returned, the node did not successfully
// join the cluster.
func (m *Memberlist) Join(existing []string) (int, error) {
	// Attempt to join any of them
	numSuccess := 0
	var retErr error
	for _, exist := range existing {
		addrs, port, err := m.resolveAddr(exist)
		if err != nil {
			m.logger.Printf("[WARN] memberlist: Failed to resolve %s: %v", exist, err)
			retErr = err
			continue
		}

		for _, addr := range addrs {
			if err := m.pushPullNode(addr, port, true); err != nil {
				retErr = err
				continue
			}
			numSuccess++
		}

	}

	if numSuccess > 0 {
		retErr = nil
	}

	return numSuccess, retErr
}

// resolveAddr is used to resolve the address into an address,
// port, and error. If no port is given, use the default
func (m *Memberlist) resolveAddr(hostStr string) ([][]byte, uint16, error) {
	ips := make([][]byte, 0)
	port := uint16(0)
	host, sport, err := net.SplitHostPort(hostStr)
	if ae, ok := err.(*net.AddrError); ok && ae.Err == "missing port in address" {
		// error, port missing - we can solve this
		port = uint16(m.config.BindPort)
		host = hostStr
	} else if err != nil {
		// error, but not missing port
		return ips, port, err
	} else if lport, err := strconv.ParseUint(sport, 10, 16); err != nil {
		// error, when parsing port
		return ips, port, err
	} else {
		// no error
		port = uint16(lport)
	}

	// Get the addresses that hostPort might resolve to
	// ResolveTcpAddr requres ipv6 brackets to separate
	// port numbers whereas ParseIP doesn't, but luckily
	// SplitHostPort takes care of the brackets
	if ip := net.ParseIP(host); ip == nil {
		if pre, err := net.LookupIP(host); err == nil {
			for _, ip := range pre {
				ips = append(ips, ip)
			}
		} else {
			return ips, port, err
		}
	} else {
		ips = append(ips, ip)
	}

	return ips, port, nil
}

// setAlive is used to mark this node as being alive. This is the same
// as if we received an alive notification our own network channel for
// ourself.
func (m *Memberlist) setAlive() error {
	var advertiseAddr []byte
	var advertisePort int
	if m.config.AdvertiseAddr != "" {
		// If AdvertiseAddr is not empty, then advertise
		// the given address and port.
		ip := net.ParseIP(m.config.AdvertiseAddr)
		if ip == nil {
			return fmt.Errorf("Failed to parse advertise address!")
		}

		// Ensure IPv4 conversion if necessary
		if ip4 := ip.To4(); ip4 != nil {
			ip = ip4
		}

		advertiseAddr = ip
		advertisePort = m.config.AdvertisePort
	} else {
		if m.config.BindAddr == "0.0.0.0" {
			// Otherwise, if we're not bound to a specific IP,
			//let's list the interfaces on this machine and use
			// the first private IP we find.
			addresses, err := net.InterfaceAddrs()
			if err != nil {
				return fmt.Errorf("Failed to get interface addresses! Err: %v", err)
			}

			// Find private IPv4 address
			for _, rawAddr := range addresses {
				var ip net.IP
				switch addr := rawAddr.(type) {
				case *net.IPAddr:
					ip = addr.IP
				case *net.IPNet:
					ip = addr.IP
				default:
					continue
				}

				if ip.To4() == nil {
					continue
				}
				if !IsPrivateIP(ip.String()) {
					continue
				}

				advertiseAddr = ip
				break
			}

			// Failed to find private IP, error
			if advertiseAddr == nil {
				return fmt.Errorf("No private IP address found, and explicit IP not provided")
			}

		} else {
			// Use the IP that we're bound to.
			addr := m.tcpListener.Addr().(*net.TCPAddr)
			advertiseAddr = addr.IP
		}

		// Use the port we are bound to.
		advertisePort = m.tcpListener.Addr().(*net.TCPAddr).Port
	}

	// Check if this is a public address without encryption
	addrStr := net.IP(advertiseAddr).String()
	if !IsPrivateIP(addrStr) && !isLoopbackIP(addrStr) && !m.config.EncryptionEnabled() {
		m.logger.Printf("[WARN] memberlist: Binding to public address without encryption!")
	}

	// Get the node meta data
	var meta []byte
	if m.config.Delegate != nil {
		meta = m.config.Delegate.NodeMeta(MetaMaxSize)
		if len(meta) > MetaMaxSize {
			panic("Node meta data provided is longer than the limit")
		}
	}

	a := alive{
		Incarnation: m.nextIncarnation(),
		Node:        m.config.Name,
		Addr:        advertiseAddr,
		Port:        uint16(advertisePort),
		Meta:        meta,
		Vsn: []uint8{
			ProtocolVersionMin, ProtocolVersionMax, m.config.ProtocolVersion,
			m.config.DelegateProtocolMin, m.config.DelegateProtocolMax,
			m.config.DelegateProtocolVersion,
		},
	}
	m.aliveNode(&a, nil, true)

	return nil
}

// LocalNode is used to return the local Node
func (m *Memberlist) LocalNode() *Node {
	m.nodeLock.RLock()
	defer m.nodeLock.RUnlock()
	state := m.nodeMap[m.config.Name]
	return &state.Node
}

// UpdateNode is used to trigger re-advertising the local node. This is
// primarily used with a Delegate to support dynamic updates to the local
// meta data.  This will block until the update message is successfully
// broadcasted to a member of the cluster, if any exist or until a specified
// timeout is reached.
func (m *Memberlist) UpdateNode(timeout time.Duration) error {
	// Get the node meta data
	var meta []byte
	if m.config.Delegate != nil {
		meta = m.config.Delegate.NodeMeta(MetaMaxSize)
		if len(meta) > MetaMaxSize {
			panic("Node meta data provided is longer than the limit")
		}
	}

	// Get the existing node
	m.nodeLock.RLock()
	state := m.nodeMap[m.config.Name]
	m.nodeLock.RUnlock()

	// Format a new alive message
	a := alive{
		Incarnation: m.nextIncarnation(),
		Node:        m.config.Name,
		Addr:        state.Addr,
		Port:        state.Port,
		Meta:        meta,
		Vsn: []uint8{
			ProtocolVersionMin, ProtocolVersionMax, m.config.ProtocolVersion,
			m.config.DelegateProtocolMin, m.config.DelegateProtocolMax,
			m.config.DelegateProtocolVersion,
		},
	}
	notifyCh := make(chan struct{})
	m.aliveNode(&a, notifyCh, true)

	// Wait for the broadcast or a timeout
	if m.anyAlive() {
		var timeoutCh <-chan time.Time
		if timeout > 0 {
			timeoutCh = time.After(timeout)
		}
		select {
		case <-notifyCh:
		case <-timeoutCh:
			return fmt.Errorf("timeout waiting for update broadcast")
		}
	}
	return nil
}

// SendTo is used to directly send a message to another node, without
// the use of the gossip mechanism. This will encode the message as a
// user-data message, which a delegate will receive through NotifyMsg
// The actual data is transmitted over UDP, which means this is a
// best-effort transmission mechanism, and the maximum size of the
// message is the size of a single UDP datagram, after compression.
// This method is DEPRECATED in favor or SendToUDP
func (m *Memberlist) SendTo(to net.Addr, msg []byte) error {
	// Encode as a user message
	buf := make([]byte, 1, len(msg)+1)
	buf[0] = byte(userMsg)
	buf = append(buf, msg...)

	// Send the message
	return m.rawSendMsgUDP(to, buf)
}

// SendToUDP is used to directly send a message to another node, without
// the use of the gossip mechanism. This will encode the message as a
// user-data message, which a delegate will receive through NotifyMsg
// The actual data is transmitted over UDP, which means this is a
// best-effort transmission mechanism, and the maximum size of the
// message is the size of a single UDP datagram, after compression
func (m *Memberlist) SendToUDP(to *Node, msg []byte) error {
	// Encode as a user message
	buf := make([]byte, 1, len(msg)+1)
	buf[0] = byte(userMsg)
	buf = append(buf, msg...)

	// Send the message
	destAddr := &net.UDPAddr{IP: to.Addr, Port: int(to.Port)}
	return m.rawSendMsgUDP(destAddr, buf)
}

// SendToTCP is used to directly send a message to another node, without
// the use of the gossip mechanism. This will encode the message as a
// user-data message, which a delegate will receive through NotifyMsg
// The actual data is transmitted over TCP, which means delivery
// is guaranteed if no error is returned. There is no limit
// to the size of the message
func (m *Memberlist) SendToTCP(to *Node, msg []byte) error {
	// Send the message
	destAddr := &net.TCPAddr{IP: to.Addr, Port: int(to.Port)}
	return m.sendTCPUserMsg(destAddr, msg)
}

// Members returns a list of all known live nodes. The node structures
// returned must not be modified. If you wish to modify a Node, make a
// copy first.
func (m *Memberlist) Members() []*Node {
	m.nodeLock.RLock()
	defer m.nodeLock.RUnlock()

	nodes := make([]*Node, 0, len(m.nodes))
	for _, n := range m.nodes {
		if n.State != stateDead {
			nodes = append(nodes, &n.Node)
		}
	}

	return nodes
}

// NumMembers returns the number of alive nodes currently known. Between
// the time of calling this and calling Members, the number of alive nodes
// may have changed, so this shouldn't be used to determine how many
// members will be returned by Members.
func (m *Memberlist) NumMembers() (alive int) {
	m.nodeLock.RLock()
	defer m.nodeLock.RUnlock()

	for _, n := range m.nodes {
		if n.State != stateDead {
			alive++
		}
	}

	return
}

// Leave will broadcast a leave message but will not shutdown the background
// listeners, meaning the node will continue participating in gossip and state
// updates.
//
// This will block until the leave message is successfully broadcasted to
// a member of the cluster, if any exist or until a specified timeout
// is reached.
//
// This method is safe to call multiple times, but must not be called
// after the cluster is already shut down.
func (m *Memberlist) Leave(timeout time.Duration) error {
	m.nodeLock.Lock()
	// We can't defer m.nodeLock.Unlock() because m.deadNode will also try to
	// acquire a lock so we need to Unlock before that.

	if m.shutdown {
		m.nodeLock.Unlock()
		panic("leave after shutdown")
	}

	if !m.leave {
		m.leave = true

		state, ok := m.nodeMap[m.config.Name]
		m.nodeLock.Unlock()
		if !ok {
			m.logger.Printf("[WARN] memberlist: Leave but we're not in the node map.")
			return nil
		}

		d := dead{
			Incarnation: state.Incarnation,
			Node:        state.Name,
		}
		m.deadNode(&d)

		// Block until the broadcast goes out
		if m.anyAlive() {
			var timeoutCh <-chan time.Time
			if timeout > 0 {
				timeoutCh = time.After(timeout)
			}
			select {
			case <-m.leaveBroadcast:
			case <-timeoutCh:
				return fmt.Errorf("timeout waiting for leave broadcast")
			}
		}
	} else {
		m.nodeLock.Unlock()
	}

	return nil
}

// Check for any other alive node.
func (m *Memberlist) anyAlive() bool {
	m.nodeLock.RLock()
	defer m.nodeLock.RUnlock()
	for _, n := range m.nodes {
		if n.State != stateDead && n.Name != m.config.Name {
			return true
		}
	}
	return false
}

// ProtocolVersion returns the protocol version currently in use by
// this memberlist.
func (m *Memberlist) ProtocolVersion() uint8 {
	// NOTE: This method exists so that in the future we can control
	// any locking if necessary, if we change the protocol version at
	// runtime, etc.
	return m.config.ProtocolVersion
}

// Shutdown will stop any background maintanence of network activity
// for this memberlist, causing it to appear "dead". A leave message
// will not be broadcasted prior, so the cluster being left will have
// to detect this node's shutdown using probing. If you wish to more
// gracefully exit the cluster, call Leave prior to shutting down.
//
// This method is safe to call multiple times.
func (m *Memberlist) Shutdown() error {
	m.nodeLock.Lock()
	defer m.nodeLock.Unlock()

	if m.shutdown {
		return nil
	}

	m.shutdown = true
	close(m.shutdownCh)
	m.deschedule()
	m.udpListener.Close()
	m.tcpListener.Close()
	return nil
}
