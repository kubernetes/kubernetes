package netlink

import (
	"iter"
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

	// receiveMu serializes concurrent Receive and ReceiveIter calls to prevent
	// races in multi-part message handling and the peek/allocate logic in
	// conn_linux.go. It is separate from mu so that Send can proceed
	// concurrently with Receive.
	receiveMu sync.Mutex

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
	ReceiveIter() iter.Seq2[Message, error]
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

	// Serialize concurrent Receive calls. See receiveMu for details.
	c.receiveMu.Lock()
	defer c.receiveMu.Unlock()

	return c.lockedReceive()
}

// ReceiveIter returns an iterator which can be used to receive messages from
// netlink. Just like Receive, multi-part messages are handled transparently and
// netlink errors are returned as errors from the iterator.
//
// If the iteration is stopped before all messages have been read and the
// response is multi-part, the remaining messages will be discarded.
func (c *Conn) ReceiveIter() iter.Seq2[Message, error] {
	return func(yield func(Message, error) bool) {
		// Wait for any concurrent calls to Execute to finish before proceeding.
		c.mu.RLock()
		defer c.mu.RUnlock()

		// Serialize concurrent ReceiveIter calls. See receiveMu for details.
		c.receiveMu.Lock()
		defer c.receiveMu.Unlock()

		for msg, err := range c.lockedReceiveIter() {
			if err != nil {
				c.debug(func(d *debugger) {
					d.debugf(1, "recv: err: %v", err)
				})
				yield(Message{}, err)
				return
			}

			c.debug(func(d *debugger) {
				d.debugf(1, "recv: %+v", msg)
			})
			if !yield(msg, nil) {
				return
			}
		}
	}
}

// lockedReceive implements Receive, but must be called with c.mu acquired for reading.
// We rely on the kernel to deal with concurrent reads and writes to the netlink
// socket itself.
func (c *Conn) lockedReceive() ([]Message, error) {
	var msgs []Message

	for m, err := range c.lockedReceiveIter() {
		if err != nil {
			c.debug(func(d *debugger) {
				d.debugf(1, "recv: err: %v", err)
			})
			return nil, err
		}

		c.debug(func(d *debugger) {
			d.debugf(1, "recv: %+v", m)
		})

		msgs = append(msgs, m)
	}

	return msgs, nil
}

// lockedReceiveIter returns an iterator which can be used to receive messages
// from netlink, but must be called with c.mu acquired for the duration of the
// iteration.
func (c *Conn) lockedReceiveIter() iter.Seq2[Message, error] {
	return func(yield func(Message, error) bool) {
		// NB: All non-nil errors returned from this function *must* be of type
		// OpError in order to maintain the appropriate contract with callers of
		// this package.
		//
		// This contract also applies to functions called within this function,
		// such as checkMessage.

		var more, stopped bool
		// send is a helper function to prevent yielding messages after the user
		// has stopped iterating
		var send = func(m Message, err error) {
			if stopped {
				return
			}
			if !yield(m, err) {
				stopped = true
			}
		}

		for {
			for m, err := range c.sock.ReceiveIter() {
				if err != nil {
					send(Message{}, newOpError("receive", err))
					return
				}

				if err := checkMessage(m); err != nil {
					send(Message{}, err)
					return
				}

				// Exit early if we encounter a multi-part done message.
				// This should be safe to do since messages of type Done should always
				// be the last message in a datagram.
				if m.Header.Type == Done && m.Header.Flags&Multi != 0 {
					return
				}

				if m.Header.Flags&Multi != 0 {
					more = true
				}

				send(m, nil)
				if stopped && !more {
					// The user has stopped iterating and there are no more messages
					// to read.
					return
				}
			}

			if !more {
				return
			}
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

// A bufferedSocket is a Socket that supports getting & setting connection
// buffer sizes.
type bufferedSocket interface {
	Socket
	SetReadBuffer(bytes int) error
	SetWriteBuffer(bytes int) error
	ReadBuffer() (int, error)
	WriteBuffer() (int, error)
}

// SetReadBuffer sets the size of the operating system's receive buffer
// associated with the Conn.
func (c *Conn) SetReadBuffer(bytes int) error {
	conn, ok := c.sock.(bufferedSocket)
	if !ok {
		return notSupported("set-read-buffer")
	}

	return newOpError("set-read-buffer", conn.SetReadBuffer(bytes))
}

// SetWriteBuffer sets the size of the operating system's transmit buffer
// associated with the Conn.
func (c *Conn) SetWriteBuffer(bytes int) error {
	conn, ok := c.sock.(bufferedSocket)
	if !ok {
		return notSupported("set-write-buffer")
	}

	return newOpError("set-write-buffer", conn.SetWriteBuffer(bytes))
}

// ReadBuffer reads the size of the operating system's receive buffer
// associated with the Conn.
func (c *Conn) ReadBuffer() (int, error) {
	conn, ok := c.sock.(bufferedSocket)
	if !ok {
		return 0, notSupported("get-read-buffer")
	}

	buff, err := conn.ReadBuffer()
	if err != nil {
		return 0, newOpError("get-read-buffer", err)
	}
	return buff, nil
}

// WriteBuffer reads the size of the operating system's transmit buffer
// associated with the Conn.
func (c *Conn) WriteBuffer() (int, error) {
	conn, ok := c.sock.(bufferedSocket)
	if !ok {
		return 0, notSupported("get-write-buffer")
	}

	buff, err := conn.WriteBuffer()
	if err != nil {
		return 0, newOpError("get-write-buffer", err)
	}

	return buff, nil
}

// PID returns the PID associated with the Conn. It is also known as
// the port ID in netlink terminology.
// https://docs.kernel.org/userspace-api/netlink/intro.html#nlmsg-pid
func (c *Conn) PID() uint32 {
	return c.pid
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

	// MessageBufferSize specifies a fixed buffer size for receiving netlink
	// messages. When set, the connection will reuse a single pre-allocated
	// buffer of this size instead of peeking at each message to determine
	// the exact size needed.
	//
	// This is useful for high-throughput applications where the overhead of
	// peeking at each message is undesirable and the maximum message size
	// is known in advance.
	//
	// If set to 0 (the default), the connection will peek at the upcoming
	// message before allocating a buffer for it.
	//
	// Note: this is not the same as the kernel socket receive buffer which
	// can be configured using SetReadBuffer. MessageBufferSize only controls
	// the userspace buffer passed to recvmsg.
	MessageBufferSize int
}
