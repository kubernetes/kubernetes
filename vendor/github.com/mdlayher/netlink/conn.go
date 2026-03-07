package netlink

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/net/bpf"
)

// A Conn is a connection to netlink.  A Conn can be used to send and
// receives messages to and from netlink.
//
// A Conn is safe for concurrent use, but to avoid contention in
// high-throughput applications, the caller should almost certainly create a
// pool of Conns and distribute them among workers.
//
// A Conn is capable of manipulating netlink subsystems from within a specific
// Linux network namespace, but special care must be taken when doing so. See
// the documentation of Config for details.
type Conn struct {
	// Atomics must come first.
	//
	// seq is an atomically incremented integer used to provide sequence
	// numbers when Conn.Send is called.
	seq uint32

	// mu serializes access to the netlink socket for the request/response
	// transaction within Execute.
	mu sync.RWMutex

	// sock is the operating system-specific implementation of
	// a netlink sockets connection.
	sock Socket

	// pid is the PID assigned by netlink.
	pid uint32

	// d provides debugging capabilities for a Conn if not nil.
	d *debugger
}

// A Socket is an operating-system specific implementation of netlink
// sockets used by Conn.
//
// Deprecated: the intent of Socket was to provide an abstraction layer for
// testing, but this abstraction is awkward to use properly and disables much of
// the functionality of the Conn type. Do not use.
type Socket interface {
	Close() error
	Send(m Message) error
	SendMessages(m []Message) error
	Receive() ([]Message, error)
}

// Dial dials a connection to netlink, using the specified netlink family.
// Config specifies optional configuration for Conn. If config is nil, a default
// configuration will be used.
func Dial(family int, config *Config) (*Conn, error) {
	// TODO(mdlayher): plumb in netlink.OpError wrapping?

	// Use OS-specific dial() to create Socket.
	c, pid, err := dial(family, config)
	if err != nil {
		return nil, err
	}

	return NewConn(c, pid), nil
}

// NewConn creates a Conn using the specified Socket and PID for netlink
// communications.
//
// NewConn is primarily useful for tests. Most applications should use
// Dial instead.
func NewConn(sock Socket, pid uint32) *Conn {
	// Seed the sequence number using a random number generator.
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	seq := r.Uint32()

	// Configure a debugger if arguments are set.
	var d *debugger
	if len(debugArgs) > 0 {
		d = newDebugger(debugArgs)
	}

	return &Conn{
		seq:  seq,
		sock: sock,
		pid:  pid,
		d:    d,
	}
}

// debug executes fn with the debugger if the debugger is not nil.
func (c *Conn) debug(fn func(d *debugger)) {
	if c.d == nil {
		return
	}

	fn(c.d)
}

// Close closes the connection and unblocks any pending read operations.
func (c *Conn) Close() error {
	// Close does not acquire a lock because it must be able to interrupt any
	// blocked system calls, such as when Receive is waiting on a multicast
	// group message.
	//
	// We rely on the kernel to deal with concurrent operations to the netlink
	// socket itself.
	return newOpError("close", c.sock.Close())
}

// Execute sends a single Message to netlink using Send, receives one or more
// replies using Receive, and then checks the validity of the replies against
// the request using Validate.
//
// Execute acquires a lock for the duration of the function call which blocks
// concurrent calls to Send, SendMessages, and Receive, in order to ensure
// consistency between netlink request/reply messages.
//
// See the documentation of Send, Receive, and Validate for details about
// each function.
func (c *Conn) Execute(m Message) ([]Message, error) {
	// Acquire the write lock and invoke the internal implementations of Send
	// and Receive which require the lock already be held.
	c.mu.Lock()
	defer c.mu.Unlock()

	req, err := c.lockedSend(m)
	if err != nil {
		return nil, err
	}

	res, err := c.lockedReceive()
	if err != nil {
		return nil, err
	}

	if err := Validate(req, res); err != nil {
		return nil, err
	}

	return res, nil
}

// SendMessages sends multiple Messages to netlink. The handling of
// a Header's Length, Sequence and PID fields is the same as when
// calling Send.
func (c *Conn) SendMessages(msgs []Message) ([]Message, error) {
	// Wait for any concurrent calls to Execute to finish before proceeding.
	c.mu.RLock()
	defer c.mu.RUnlock()

	for i := range msgs {
		c.fixMsg(&msgs[i], nlmsgLength(len(msgs[i].Data)))
	}

	c.debug(func(d *debugger) {
		for _, m := range msgs {
			d.debugf(1, "send msgs: %+v", m)
		}
	})

	if err := c.sock.SendMessages(msgs); err != nil {
		c.debug(func(d *debugger) {
			d.debugf(1, "send msgs: err: %v", err)
		})

		return nil, newOpError("send-messages", err)
	}

	return msgs, nil
}

// Send sends a single Message to netlink.  In most cases, a Header's Length,
// Sequence, and PID fields should be set to 0, so they can be populated
// automatically before the Message is sent.  On success, Send returns a copy
// of the Message with all parameters populated, for later validation.
//
// If Header.Length is 0, it will be automatically populated using the
// correct length for the Message, including its payload.
//
// If Header.Sequence is 0, it will be automatically populated using the
// next sequence number for this connection.
//
// If Header.PID is 0, it will be automatically populated using a PID
// assigned by netlink.
func (c *Conn) Send(m Message) (Message, error) {
	// Wait for any concurrent calls to Execute to finish before proceeding.
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.lockedSend(m)
}

// lockedSend implements Send, but must be called with c.mu acquired for reading.
// We rely on the kernel to deal with concurrent reads and writes to the netlink
// socket itself.
func (c *Conn) lockedSend(m Message) (Message, error) {
	c.fixMsg(&m, nlmsgLength(len(m.Data)))

	c.debug(func(d *debugger) {
		d.debugf(1, "send: %+v", m)
	})

	if err := c.sock.Send(m); err != nil {
		c.debug(func(d *debugger) {
			d.debugf(1, "send: err: %v", err)
		})

		return Message{}, newOpError("send", err)
	}

	return m, nil
}

// Receive receives one or more messages from netlink.  Multi-part messages are
// handled transparently and returned as a single slice of Messages, with the
// final empty "multi-part done" message removed.
//
// If any of the messages indicate a netlink error, that error will be returned.
func (c *Conn) Receive() ([]Message, error) {
	// Wait for any concurrent calls to Execute to finish before proceeding.
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.lockedReceive()
}

// lockedReceive implements Receive, but must be called with c.mu acquired for reading.
// We rely on the kernel to deal with concurrent reads and writes to the netlink
// socket itself.
func (c *Conn) lockedReceive() ([]Message, error) {
	msgs, err := c.receive()
	if err != nil {
		c.debug(func(d *debugger) {
			d.debugf(1, "recv: err: %v", err)
		})

		return nil, err
	}

	c.debug(func(d *debugger) {
		for _, m := range msgs {
			d.debugf(1, "recv: %+v", m)
		}
	})

	// When using nltest, it's possible for zero messages to be returned by receive.
	if len(msgs) == 0 {
		return msgs, nil
	}

	// Trim the final message with multi-part done indicator if
	// present.
	if m := msgs[len(msgs)-1]; m.Header.Flags&Multi != 0 && m.Header.Type == Done {
		return msgs[:len(msgs)-1], nil
	}

	return msgs, nil
}

// receive is the internal implementation of Conn.Receive, which can be called
// recursively to handle multi-part messages.
func (c *Conn) receive() ([]Message, error) {
	// NB: All non-nil errors returned from this function *must* be of type
	// OpError in order to maintain the appropriate contract with callers of
	// this package.
	//
	// This contract also applies to functions called within this function,
	// such as checkMessage.

	var res []Message
	for {
		msgs, err := c.sock.Receive()
		if err != nil {
			return nil, newOpError("receive", err)
		}

		// If this message is multi-part, we will need to continue looping to
		// drain all the messages from the socket.
		var multi bool

		for _, m := range msgs {
			if err := checkMessage(m); err != nil {
				return nil, err
			}

			// Does this message indicate a multi-part message?
			if m.Header.Flags&Multi == 0 {
				// No, check the next messages.
				continue
			}

			// Does this message indicate the last message in a series of
			// multi-part messages from a single read?
			multi = m.Header.Type != Done
		}

		res = append(res, msgs...)

		if !multi {
			// No more messages coming.
			return res, nil
		}
	}
}

// A groupJoinLeaver is a Socket that supports joining and leaving
// netlink multicast groups.
type groupJoinLeaver interface {
	Socket
	JoinGroup(group uint32) error
	LeaveGroup(group uint32) error
}

// JoinGroup joins a netlink multicast group by its ID.
func (c *Conn) JoinGroup(group uint32) error {
	conn, ok := c.sock.(groupJoinLeaver)
	if !ok {
		return notSupported("join-group")
	}

	return newOpError("join-group", conn.JoinGroup(group))
}

// LeaveGroup leaves a netlink multicast group by its ID.
func (c *Conn) LeaveGroup(group uint32) error {
	conn, ok := c.sock.(groupJoinLeaver)
	if !ok {
		return notSupported("leave-group")
	}

	return newOpError("leave-group", conn.LeaveGroup(group))
}

// A bpfSetter is a Socket that supports setting and removing BPF filters.
type bpfSetter interface {
	Socket
	bpf.Setter
	RemoveBPF() error
}

// SetBPF attaches an assembled BPF program to a Conn.
func (c *Conn) SetBPF(filter []bpf.RawInstruction) error {
	conn, ok := c.sock.(bpfSetter)
	if !ok {
		return notSupported("set-bpf")
	}

	return newOpError("set-bpf", conn.SetBPF(filter))
}

// RemoveBPF removes a BPF filter from a Conn.
func (c *Conn) RemoveBPF() error {
	conn, ok := c.sock.(bpfSetter)
	if !ok {
		return notSupported("remove-bpf")
	}

	return newOpError("remove-bpf", conn.RemoveBPF())
}

// A deadlineSetter is a Socket that supports setting deadlines.
type deadlineSetter interface {
	Socket
	SetDeadline(time.Time) error
	SetReadDeadline(time.Time) error
	SetWriteDeadline(time.Time) error
}

// SetDeadline sets the read and write deadlines associated with the connection.
func (c *Conn) SetDeadline(t time.Time) error {
	conn, ok := c.sock.(deadlineSetter)
	if !ok {
		return notSupported("set-deadline")
	}

	return newOpError("set-deadline", conn.SetDeadline(t))
}

// SetReadDeadline sets the read deadline associated with the connection.
func (c *Conn) SetReadDeadline(t time.Time) error {
	conn, ok := c.sock.(deadlineSetter)
	if !ok {
		return notSupported("set-read-deadline")
	}

	return newOpError("set-read-deadline", conn.SetReadDeadline(t))
}

// SetWriteDeadline sets the write deadline associated with the connection.
func (c *Conn) SetWriteDeadline(t time.Time) error {
	conn, ok := c.sock.(deadlineSetter)
	if !ok {
		return notSupported("set-write-deadline")
	}

	return newOpError("set-write-deadline", conn.SetWriteDeadline(t))
}

// A ConnOption is a boolean option that may be set for a Conn.
type ConnOption int

// Possible ConnOption values.  These constants are equivalent to the Linux
// setsockopt boolean options for netlink sockets.
const (
	PacketInfo ConnOption = iota
	BroadcastError
	NoENOBUFS
	ListenAllNSID
	CapAcknowledge
	ExtendedAcknowledge
	GetStrictCheck
)

// An optionSetter is a Socket that supports setting netlink options.
type optionSetter interface {
	Socket
	SetOption(option ConnOption, enable bool) error
}

// SetOption enables or disables a netlink socket option for the Conn.
func (c *Conn) SetOption(option ConnOption, enable bool) error {
	conn, ok := c.sock.(optionSetter)
	if !ok {
		return notSupported("set-option")
	}

	return newOpError("set-option", conn.SetOption(option, enable))
}

// A bufferSetter is a Socket that supports setting connection buffer sizes.
type bufferSetter interface {
	Socket
	SetReadBuffer(bytes int) error
	SetWriteBuffer(bytes int) error
}

// SetReadBuffer sets the size of the operating system's receive buffer
// associated with the Conn.
func (c *Conn) SetReadBuffer(bytes int) error {
	conn, ok := c.sock.(bufferSetter)
	if !ok {
		return notSupported("set-read-buffer")
	}

	return newOpError("set-read-buffer", conn.SetReadBuffer(bytes))
}

// SetWriteBuffer sets the size of the operating system's transmit buffer
// associated with the Conn.
func (c *Conn) SetWriteBuffer(bytes int) error {
	conn, ok := c.sock.(bufferSetter)
	if !ok {
		return notSupported("set-write-buffer")
	}

	return newOpError("set-write-buffer", conn.SetWriteBuffer(bytes))
}

// A syscallConner is a Socket that supports syscall.Conn.
type syscallConner interface {
	Socket
	syscall.Conn
}

var _ syscall.Conn = &Conn{}

// SyscallConn returns a raw network connection. This implements the
// syscall.Conn interface.
//
// SyscallConn is intended for advanced use cases, such as getting and setting
// arbitrary socket options using the netlink socket's file descriptor.
//
// Once invoked, it is the caller's responsibility to ensure that operations
// performed using Conn and the syscall.RawConn do not conflict with
// each other.
func (c *Conn) SyscallConn() (syscall.RawConn, error) {
	sc, ok := c.sock.(syscallConner)
	if !ok {
		return nil, notSupported("syscall-conn")
	}

	// TODO(mdlayher): mutex or similar to enforce syscall.RawConn contract of
	// FD remaining valid for duration of calls?

	return sc.SyscallConn()
}

// fixMsg updates the fields of m using the logic specified in Send.
func (c *Conn) fixMsg(m *Message, ml int) {
	if m.Header.Length == 0 {
		m.Header.Length = uint32(nlmsgAlign(ml))
	}

	if m.Header.Sequence == 0 {
		m.Header.Sequence = c.nextSequence()
	}

	if m.Header.PID == 0 {
		m.Header.PID = c.pid
	}
}

// nextSequence atomically increments Conn's sequence number and returns
// the incremented value.
func (c *Conn) nextSequence() uint32 {
	return atomic.AddUint32(&c.seq, 1)
}

// Validate validates one or more reply Messages against a request Message,
// ensuring that they contain matching sequence numbers and PIDs.
func Validate(request Message, replies []Message) error {
	for _, m := range replies {
		// Check for mismatched sequence, unless:
		//   - request had no sequence, meaning we are probably validating
		//     a multicast reply
		if m.Header.Sequence != request.Header.Sequence && request.Header.Sequence != 0 {
			return newOpError("validate", errMismatchedSequence)
		}

		// Check for mismatched PID, unless:
		//   - request had no PID, meaning we are either:
		//     - validating a multicast reply
		//     - netlink has not yet assigned us a PID
		//   - response had no PID, meaning it's from the kernel as a multicast reply
		if m.Header.PID != request.Header.PID && request.Header.PID != 0 && m.Header.PID != 0 {
			return newOpError("validate", errMismatchedPID)
		}
	}

	return nil
}

// Config contains options for a Conn.
type Config struct {
	// Groups is a bitmask which specifies multicast groups. If set to 0,
	// no multicast group subscriptions will be made.
	Groups uint32

	// NetNS specifies the network namespace the Conn will operate in.
	//
	// If set (non-zero), Conn will enter the specified network namespace and
	// an error will occur in Dial if the operation fails.
	//
	// If not set (zero), a best-effort attempt will be made to enter the
	// network namespace of the calling thread: this means that any changes made
	// to the calling thread's network namespace will also be reflected in Conn.
	// If this operation fails (due to lack of permissions or because network
	// namespaces are disabled by kernel configuration), Dial will not return
	// an error, and the Conn will operate in the default network namespace of
	// the process. This enables non-privileged use of Conn in applications
	// which do not require elevated privileges.
	//
	// Entering a network namespace is a privileged operation (root or
	// CAP_SYS_ADMIN are required), and most applications should leave this set
	// to 0.
	NetNS int

	// DisableNSLockThread is a no-op.
	//
	// Deprecated: internal changes have made this option obsolete and it has no
	// effect. Do not use.
	DisableNSLockThread bool

	// PID specifies the port ID used to bind the netlink socket. If set to 0,
	// the kernel will assign a port ID on the caller's behalf.
	//
	// Most callers should leave this field set to 0. This option is intended
	// for advanced use cases where the kernel expects a fixed unicast address
	// destination for netlink messages.
	PID uint32

	// Strict applies a more strict default set of options to the Conn,
	// including:
	//   - ExtendedAcknowledge: true
	//     - provides more useful error messages when supported by the kernel
	//   - GetStrictCheck: true
	//     - more strictly enforces request validation for some families such
	//       as rtnetlink which were historically misused
	//
	// If any of the options specified by Strict cannot be configured due to an
	// outdated kernel or similar, an error will be returned.
	//
	// When possible, setting Strict to true is recommended for applications
	// running on modern Linux kernels.
	Strict bool
}
