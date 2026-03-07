package netlink

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/mdlayher/netlink/nlenc"
)

// Flags which may apply to netlink attribute types when communicating with
// certain netlink families.
const (
	Nested       uint16 = 0x8000
	NetByteOrder uint16 = 0x4000

	// attrTypeMask masks off Type bits used for the above flags.
	attrTypeMask uint16 = 0x3fff
)

// Various errors which may occur when attempting to marshal or unmarshal
// a Message to and from its binary form.
var (
	errIncorrectMessageLength = errors.New("netlink message header length incorrect")
	errShortMessage           = errors.New("not enough data to create a netlink message")
	errUnalignedMessage       = errors.New("input data is not properly aligned for netlink message")
)

// HeaderFlags specify flags which may be present in a Header.
type HeaderFlags uint16

const (
	// General netlink communication flags.

	// Request indicates a request to netlink.
	Request HeaderFlags = 1

	// Multi indicates a multi-part message, terminated by Done on the
	// last message.
	Multi HeaderFlags = 2

	// Acknowledge requests that netlink reply with an acknowledgement
	// using Error and, if needed, an error code.
	Acknowledge HeaderFlags = 4

	// Echo requests that netlink echo this request back to the sender.
	Echo HeaderFlags = 8

	// DumpInterrupted indicates that a dump was inconsistent due to a
	// sequence change.
	DumpInterrupted HeaderFlags = 16

	// DumpFiltered indicates that a dump was filtered as requested.
	DumpFiltered HeaderFlags = 32

	// Flags used to retrieve data from netlink.

	// Root requests that netlink return a complete table instead of a
	// single entry.
	Root HeaderFlags = 0x100

	// Match requests that netlink return a list of all matching entries.
	Match HeaderFlags = 0x200

	// Atomic requests that netlink send an atomic snapshot of its entries.
	// Requires CAP_NET_ADMIN or an effective UID of 0.
	Atomic HeaderFlags = 0x400

	// Dump requests that netlink return a complete list of all entries.
	Dump HeaderFlags = Root | Match

	// Flags used to create objects.

	// Replace indicates request replaces an existing matching object.
	Replace HeaderFlags = 0x100

	// Excl indicates request does not replace the object if it already exists.
	Excl HeaderFlags = 0x200

	// Create indicates request creates an object if it doesn't already exist.
	Create HeaderFlags = 0x400

	// Append indicates request adds to the end of the object list.
	Append HeaderFlags = 0x800

	// Flags for extended acknowledgements.

	// Capped indicates the size of a request was capped in an extended
	// acknowledgement.
	Capped HeaderFlags = 0x100

	// AcknowledgeTLVs indicates the presence of netlink extended
	// acknowledgement TLVs in a response.
	AcknowledgeTLVs HeaderFlags = 0x200
)

// String returns the string representation of a HeaderFlags.
func (f HeaderFlags) String() string {
	names := []string{
		"request",
		"multi",
		"acknowledge",
		"echo",
		"dumpinterrupted",
		"dumpfiltered",
	}

	var s string

	left := uint(f)

	for i, name := range names {
		if f&(1<<uint(i)) != 0 {
			if s != "" {
				s += "|"
			}

			s += name

			left ^= (1 << uint(i))
		}
	}

	if s == "" && left == 0 {
		s = "0"
	}

	if left > 0 {
		if s != "" {
			s += "|"
		}
		s += fmt.Sprintf("%#x", left)
	}

	return s
}

// HeaderType specifies the type of a Header.
type HeaderType uint16

const (
	// Noop indicates that no action was taken.
	Noop HeaderType = 0x1

	// Error indicates an error code is present, which is also used to indicate
	// success when the code is 0.
	Error HeaderType = 0x2

	// Done indicates the end of a multi-part message.
	Done HeaderType = 0x3

	// Overrun indicates that data was lost from this message.
	Overrun HeaderType = 0x4
)

// String returns the string representation of a HeaderType.
func (t HeaderType) String() string {
	switch t {
	case Noop:
		return "noop"
	case Error:
		return "error"
	case Done:
		return "done"
	case Overrun:
		return "overrun"
	default:
		return fmt.Sprintf("unknown(%d)", t)
	}
}

// NB: the memory layout of Header and Linux's syscall.NlMsgHdr must be
// exactly the same.  Cannot reorder, change data type, add, or remove fields.
// Named types of the same size (e.g. HeaderFlags is a uint16) are okay.

// A Header is a netlink header.  A Header is sent and received with each
// Message to indicate metadata regarding a Message.
type Header struct {
	// Length of a Message, including this Header.
	Length uint32

	// Contents of a Message.
	Type HeaderType

	// Flags which may be used to modify a request or response.
	Flags HeaderFlags

	// The sequence number of a Message.
	Sequence uint32

	// The port ID of the sending process.
	PID uint32
}

// A Message is a netlink message.  It contains a Header and an arbitrary
// byte payload, which may be decoded using information from the Header.
//
// Data is often populated with netlink attributes. For easy encoding and
// decoding of attributes, see the AttributeDecoder and AttributeEncoder types.
type Message struct {
	Header Header
	Data   []byte
}

// MarshalBinary marshals a Message into a byte slice.
func (m Message) MarshalBinary() ([]byte, error) {
	ml := nlmsgAlign(int(m.Header.Length))
	if ml < nlmsgHeaderLen || ml != int(m.Header.Length) {
		return nil, errIncorrectMessageLength
	}

	b := make([]byte, ml)

	nlenc.PutUint32(b[0:4], m.Header.Length)
	nlenc.PutUint16(b[4:6], uint16(m.Header.Type))
	nlenc.PutUint16(b[6:8], uint16(m.Header.Flags))
	nlenc.PutUint32(b[8:12], m.Header.Sequence)
	nlenc.PutUint32(b[12:16], m.Header.PID)
	copy(b[16:], m.Data)

	return b, nil
}

// UnmarshalBinary unmarshals the contents of a byte slice into a Message.
func (m *Message) UnmarshalBinary(b []byte) error {
	if len(b) < nlmsgHeaderLen {
		return errShortMessage
	}
	if len(b) != nlmsgAlign(len(b)) {
		return errUnalignedMessage
	}

	// Don't allow misleading length
	m.Header.Length = nlenc.Uint32(b[0:4])
	if int(m.Header.Length) != len(b) {
		return errShortMessage
	}

	m.Header.Type = HeaderType(nlenc.Uint16(b[4:6]))
	m.Header.Flags = HeaderFlags(nlenc.Uint16(b[6:8]))
	m.Header.Sequence = nlenc.Uint32(b[8:12])
	m.Header.PID = nlenc.Uint32(b[12:16])
	m.Data = b[16:]

	return nil
}

// checkMessage checks a single Message for netlink errors.
func checkMessage(m Message) error {
	// NB: All non-nil errors returned from this function *must* be of type
	// OpError in order to maintain the appropriate contract with callers of
	// this package.

	// The libnl documentation indicates that type error can
	// contain error codes:
	// https://www.infradead.org/~tgr/libnl/doc/core.html#core_errmsg.
	//
	// However, rtnetlink at least seems to also allow errors to occur at the
	// end of a multipart message with done/multi and an error number.
	var hasHeader bool
	switch {
	case m.Header.Type == Error:
		// Error code followed by nlmsghdr/ext ack attributes.
		hasHeader = true
	case m.Header.Type == Done && m.Header.Flags&Multi != 0:
		// If no data, there must be no error number so just  exit early. Some
		// of the unit tests hard-coded this but I don't actually know if this
		// case occurs in the wild.
		if len(m.Data) == 0 {
			return nil
		}

		// Done|Multi potentially followed by ext ack attributes.
	default:
		// Neither, nothing to do.
		return nil
	}

	// Errno occupies 4 bytes.
	const endErrno = 4
	if len(m.Data) < endErrno {
		return newOpError("receive", errShortErrorMessage)
	}

	c := nlenc.Int32(m.Data[:endErrno])
	if c == 0 {
		// 0 indicates no error.
		return nil
	}

	oerr := &OpError{
		Op: "receive",
		// Error code is a negative integer, convert it into an OS-specific raw
		// system call error, but do not wrap with os.NewSyscallError to signify
		// that this error was produced by a netlink message; not a system call.
		Err: newError(-1 * int(c)),
	}

	// TODO(mdlayher): investigate the Capped flag.

	if m.Header.Flags&AcknowledgeTLVs == 0 {
		// No extended acknowledgement.
		return oerr
	}

	// Flags indicate an extended acknowledgement. The type/flags combination
	// checked above determines the offset where the TLVs occur.
	var off int
	if hasHeader {
		// There is an nlmsghdr preceding the TLVs.
		if len(m.Data) < endErrno+nlmsgHeaderLen {
			return newOpError("receive", errShortErrorMessage)
		}

		// The TLVs should be at the offset indicated by the nlmsghdr.length,
		// plus the offset where the header began. But make sure the calculated
		// offset is still in-bounds.
		h := *(*Header)(unsafe.Pointer(&m.Data[endErrno : endErrno+nlmsgHeaderLen][0]))
		off = endErrno + int(h.Length)

		if len(m.Data) < off {
			return newOpError("receive", errShortErrorMessage)
		}
	} else {
		// There is no nlmsghdr preceding the TLVs, parse them directly.
		off = endErrno
	}

	ad, err := NewAttributeDecoder(m.Data[off:])
	if err != nil {
		// Malformed TLVs, just return the OpError with the info we have.
		return oerr
	}

	for ad.Next() {
		switch ad.Type() {
		case 1: // unix.NLMSGERR_ATTR_MSG
			oerr.Message = ad.String()
		case 2: // unix.NLMSGERR_ATTR_OFFS
			oerr.Offset = int(ad.Uint32())
		}
	}

	// Explicitly ignore ad.Err: malformed TLVs, just return the OpError with
	// the info we have.
	return oerr
}
