package dns

import (
	"fmt"
	"strings"
)

// PrivateRdata is an interface used for implementing "Private Use" RR types, see
// RFC 6895. This allows one to experiment with new RR types, without requesting an
// official type code. Also see dns.PrivateHandle and dns.PrivateHandleRemove.
type PrivateRdata interface {
	// String returns the text presentaton of the Rdata of the Private RR.
	String() string
	// Parse parses the Rdata of the private RR.
	Parse([]string) error
	// Pack is used when packing a private RR into a buffer.
	Pack([]byte) (int, error)
	// Unpack is used when unpacking a private RR from a buffer.
	// TODO(miek): diff. signature than Pack, see edns0.go for instance.
	Unpack([]byte) (int, error)
	// Copy copies the Rdata.
	Copy(PrivateRdata) error
	// Len returns the length in octets of the Rdata.
	Len() int
}

// PrivateRR represents an RR that uses a PrivateRdata user-defined type.
// It mocks normal RRs and implements dns.RR interface.
type PrivateRR struct {
	Hdr  RR_Header
	Data PrivateRdata
}

func mkPrivateRR(rrtype uint16) *PrivateRR {
	// Panics if RR is not an instance of PrivateRR.
	rrfunc, ok := TypeToRR[rrtype]
	if !ok {
		panic(fmt.Sprintf("dns: invalid operation with Private RR type %d", rrtype))
	}

	anyrr := rrfunc()
	rr, ok := anyrr.(*PrivateRR)
	if !ok {
		panic(fmt.Sprintf("dns: RR is not a PrivateRR, TypeToRR[%d] generator returned %T", rrtype, anyrr))
	}

	return rr
}

// Header return the RR header of r.
func (r *PrivateRR) Header() *RR_Header { return &r.Hdr }

func (r *PrivateRR) String() string { return r.Hdr.String() + r.Data.String() }

// Private len and copy parts to satisfy RR interface.
func (r *PrivateRR) len(off int, compression map[string]struct{}) int {
	l := r.Hdr.len(off, compression)
	l += r.Data.Len()
	return l
}

func (r *PrivateRR) copy() RR {
	// make new RR like this:
	rr := mkPrivateRR(r.Hdr.Rrtype)
	rr.Hdr = r.Hdr

	err := r.Data.Copy(rr.Data)
	if err != nil {
		panic("dns: got value that could not be used to copy Private rdata")
	}
	return rr
}

func (r *PrivateRR) pack(msg []byte, off int, compression compressionMap, compress bool) (int, error) {
	n, err := r.Data.Pack(msg[off:])
	if err != nil {
		return len(msg), err
	}
	off += n
	return off, nil
}

func (r *PrivateRR) unpack(msg []byte, off int) (int, error) {
	off1, err := r.Data.Unpack(msg[off:])
	off += off1
	return off, err
}

func (r *PrivateRR) parse(c *zlexer, origin, file string) *ParseError {
	var l lex
	text := make([]string, 0, 2) // could be 0..N elements, median is probably 1
Fetch:
	for {
		// TODO(miek): we could also be returning _QUOTE, this might or might not
		// be an issue (basically parsing TXT becomes hard)
		switch l, _ = c.Next(); l.value {
		case zNewline, zEOF:
			break Fetch
		case zString:
			text = append(text, l.token)
		}
	}

	err := r.Data.Parse(text)
	if err != nil {
		return &ParseError{file, err.Error(), l}
	}

	return nil
}

func (r1 *PrivateRR) isDuplicate(r2 RR) bool { return false }

// PrivateHandle registers a private resource record type. It requires
// string and numeric representation of private RR type and generator function as argument.
func PrivateHandle(rtypestr string, rtype uint16, generator func() PrivateRdata) {
	rtypestr = strings.ToUpper(rtypestr)

	TypeToRR[rtype] = func() RR { return &PrivateRR{RR_Header{}, generator()} }
	TypeToString[rtype] = rtypestr
	StringToType[rtypestr] = rtype
}

// PrivateHandleRemove removes definitions required to support private RR type.
func PrivateHandleRemove(rtype uint16) {
	rtypestr, ok := TypeToString[rtype]
	if ok {
		delete(TypeToRR, rtype)
		delete(TypeToString, rtype)
		delete(StringToType, rtypestr)
	}
}
