//go:build linux

package netlink

import (
	"context"
	"iter"
	"os"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/mdlayher/socket"
	"golang.org/x/net/bpf"
	"golang.org/x/sys/unix"
)

var _ Socket = &conn{}

// A conn is the Linux implementation of a netlink sockets connection.
type conn struct {
	s    *socket.Conn
	pool *sync.Pool
}

// dial is the entry point for Dial. dial opens a netlink socket using
// system calls, and returns its PID.
func dial(family int, config *Config) (*conn, uint32, error) {
	if config == nil {
		config = &Config{}
	}

	// Prepare the netlink socket.
	s, err := socket.Socket(
		unix.AF_NETLINK,
		unix.SOCK_RAW,
		family,
		"netlink",
		&socket.Config{NetNS: config.NetNS},
	)
	if err != nil {
		return nil, 0, err
	}

	return newConn(s, config)
}

// newConn binds a connection to netlink using the input *socket.Conn.
func newConn(s *socket.Conn, config *Config) (*conn, uint32, error) {
	if config == nil {
		config = &Config{}
	}

	addr := &unix.SockaddrNetlink{
		Family: unix.AF_NETLINK,
		Groups: config.Groups,
		Pid:    config.PID,
	}

	// Socket must be closed in the event of any system call errors, to avoid
	// leaking file descriptors.

	if err := s.Bind(addr); err != nil {
		_ = s.Close()
		return nil, 0, err
	}

	sa, err := s.Getsockname()
	if err != nil {
		_ = s.Close()
		return nil, 0, err
	}

	c := &conn{s: s}

	if config.MessageBufferSize > 0 {
		c.pool = &sync.Pool{
			New: func() any {
				b := make([]byte, config.MessageBufferSize)
				return &b
			},
		}
	}

	if config.Strict {
		// The caller has requested the strict option set. Historically we have
		// recommended checking for ENOPROTOOPT if the kernel does not support
		// the option in question, but that may result in a silent failure and
		// unexpected behavior for the user.
		//
		// Treat any error here as a fatal error, and require the caller to deal
		// with it.
		for _, o := range []ConnOption{ExtendedAcknowledge, GetStrictCheck} {
			if err := c.SetOption(o, true); err != nil {
				_ = c.Close()
				return nil, 0, err
			}
		}
	}

	return c, sa.(*unix.SockaddrNetlink).Pid, nil
}

// SendMessages serializes multiple Messages and sends them to netlink.
func (c *conn) SendMessages(messages []Message) error {
	var buf []byte
	for _, m := range messages {
		b, err := m.MarshalBinary()
		if err != nil {
			return err
		}

		buf = append(buf, b...)
	}

	sa := &unix.SockaddrNetlink{Family: unix.AF_NETLINK}
	_, err := c.s.Sendmsg(context.Background(), buf, nil, sa, 0)
	return err
}

// Send sends a single Message to netlink.
func (c *conn) Send(m Message) error {
	b, err := m.MarshalBinary()
	if err != nil {
		return err
	}

	sa := &unix.SockaddrNetlink{Family: unix.AF_NETLINK}
	_, err = c.s.Sendmsg(context.Background(), b, nil, sa, 0)
	return err
}

// Receive receives one or more Messages from netlink.
func (c *conn) Receive() ([]Message, error) {
	var msgs []Message
	for msg, err := range c.ReceiveIter() {
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}

	return msgs, nil
}

// getBuffer returns the buffer to use for receiving messages and a function to
// release it back to the pool if applicable. If the pool is not configured, a
// new buffer is allocated by peeking the size of the next message to be
// received. The buffer size is aligned to the next multiple of the alignment of a netlink
// message, to avoid panic when parsing messages from the buffer.
func (c *conn) getBuffer() ([]byte, func(), error) {
	if c.pool != nil {
		bp := c.pool.Get().(*[]byte)
		return *bp, func() { c.pool.Put(bp) }, nil
	}

	n, _, _, _, err := c.s.Recvmsg(context.Background(), nil, nil, unix.MSG_PEEK|unix.MSG_TRUNC)
	if err != nil {
		return nil, nil, err
	}

	return make([]byte, nlmsgAlign(n)), func() {}, nil
}

// ownedBuffer trims b to the aligned size of the received datagram and, when
// using pooled buffers, copies it to memory owned by the caller.
func (c *conn) ownedBuffer(b []byte, n int) []byte {
	aligned := nlmsgAlign(n)
	if c.pool == nil {
		return b[:aligned]
	}

	// A pooled buffer may not have an aligned length. Copy the received bytes
	// to an aligned buffer and let the zero-value tail bytes act as padding.
	out := make([]byte, aligned)
	copy(out, b[:n])
	return out
}

// ReceiveIter returns an iterator over Messages received from netlink.
func (c *conn) ReceiveIter() iter.Seq2[Message, error] {
	return func(yield func(Message, error) bool) {
		b, release, err := c.getBuffer()
		if err != nil {
			yield(Message{}, err)
			return
		}
		defer release()

		// Read out all available messages
		// TODO(mdlayher): deal with OOB message data if available, such as
		// when PacketInfo ConnOption is true.
		n, _, flags, _, err := c.s.Recvmsg(context.Background(), b, nil, 0)
		if err != nil {
			yield(Message{}, err)
			return
		}

		if flags&unix.MSG_TRUNC != 0 {
			// Our buffer was too small to read the entire message,
			// this should not happen since we peeked above, but if it does,
			// return an error.
			yield(Message{}, errMessageTruncated)
			return
		}

		for msg, err := range parseMessagesIter(c.ownedBuffer(b, n)) {
			if err != nil {
				yield(Message{}, err)
				return
			}

			if !yield(msg, nil) {
				return
			}
		}
	}
}

// Close closes the connection.
func (c *conn) Close() error { return c.s.Close() }

// JoinGroup joins a multicast group by ID.
func (c *conn) JoinGroup(group uint32) error {
	return c.s.SetsockoptInt(unix.SOL_NETLINK, unix.NETLINK_ADD_MEMBERSHIP, int(group))
}

// LeaveGroup leaves a multicast group by ID.
func (c *conn) LeaveGroup(group uint32) error {
	return c.s.SetsockoptInt(unix.SOL_NETLINK, unix.NETLINK_DROP_MEMBERSHIP, int(group))
}

// SetBPF attaches an assembled BPF program to a conn.
func (c *conn) SetBPF(filter []bpf.RawInstruction) error { return c.s.SetBPF(filter) }

// RemoveBPF removes a BPF filter from a conn.
func (c *conn) RemoveBPF() error { return c.s.RemoveBPF() }

// SetOption enables or disables a netlink socket option for the Conn.
func (c *conn) SetOption(option ConnOption, enable bool) error {
	o, ok := linuxOption(option)
	if !ok {
		// Return the typical Linux error for an unknown ConnOption.
		return os.NewSyscallError("setsockopt", unix.ENOPROTOOPT)
	}

	var v int
	if enable {
		v = 1
	}

	return c.s.SetsockoptInt(unix.SOL_NETLINK, o, v)
}

func (c *conn) SetDeadline(t time.Time) error      { return c.s.SetDeadline(t) }
func (c *conn) SetReadDeadline(t time.Time) error  { return c.s.SetReadDeadline(t) }
func (c *conn) SetWriteDeadline(t time.Time) error { return c.s.SetWriteDeadline(t) }

// SetReadBuffer sets the size of the operating system's receive buffer
// associated with the Conn.
func (c *conn) SetReadBuffer(bytes int) error { return c.s.SetReadBuffer(bytes) }

// SetReadBuffer sets the size of the operating system's transmit buffer
// associated with the Conn.
func (c *conn) SetWriteBuffer(bytes int) error { return c.s.SetWriteBuffer(bytes) }

// ReadBuffer reads the size of the operating system's receive buffer
// associated with the Conn.
func (c *conn) ReadBuffer() (int, error) { return c.s.ReadBuffer() }

// WriteBuffer reads the size of the operating system's transmit buffer
// associated with the Conn.
func (c *conn) WriteBuffer() (int, error) { return c.s.WriteBuffer() }

// SyscallConn returns a raw network connection.
func (c *conn) SyscallConn() (syscall.RawConn, error) { return c.s.SyscallConn() }

// linuxOption converts a ConnOption to its Linux value.
func linuxOption(o ConnOption) (int, bool) {
	switch o {
	case PacketInfo:
		return unix.NETLINK_PKTINFO, true
	case BroadcastError:
		return unix.NETLINK_BROADCAST_ERROR, true
	case NoENOBUFS:
		return unix.NETLINK_NO_ENOBUFS, true
	case ListenAllNSID:
		return unix.NETLINK_LISTEN_ALL_NSID, true
	case CapAcknowledge:
		return unix.NETLINK_CAP_ACK, true
	case ExtendedAcknowledge:
		return unix.NETLINK_EXT_ACK, true
	case GetStrictCheck:
		return unix.NETLINK_GET_STRICT_CHK, true
	default:
		return 0, false
	}
}

// sysToHeader converts a syscall.NlMsghdr to a Header.
func sysToHeader(r syscall.NlMsghdr) Header {
	// NB: the memory layout of Header and syscall.NlMsgHdr must be
	// exactly the same for this unsafe cast to work
	return *(*Header)(unsafe.Pointer(&r))
}

// newError converts an error number from netlink into the appropriate
// system call error for Linux.
func newError(errno int) error {
	return syscall.Errno(errno)
}
