package dns

import "strconv"

const (
	year68     = 1 << 31 // For RFC1982 (Serial Arithmetic) calculations in 32 bits.
	defaultTtl = 3600    // Default internal TTL.

	DefaultMsgSize = 4096  // DefaultMsgSize is the standard default for messages larger than 512 bytes.
	MinMsgSize     = 512   // MinMsgSize is the minimal size of a DNS packet.
	MaxMsgSize     = 65535 // MaxMsgSize is the largest possible DNS packet.
)

// Error represents a DNS error.
type Error struct{ err string }

func (e *Error) Error() string {
	if e == nil {
		return "dns: <nil>"
	}
	return "dns: " + e.err
}

// An RR represents a resource record.
type RR interface {
	// Header returns the header of an resource record. The header contains
	// everything up to the rdata.
	Header() *RR_Header
	// String returns the text representation of the resource record.
	String() string

	// copy returns a copy of the RR
	copy() RR
	// len returns the length (in octets) of the uncompressed RR in wire format.
	len() int
	// pack packs an RR into wire format.
	pack([]byte, int, map[string]int, bool) (int, error)
}

// RR_Header is the header all DNS resource records share.
type RR_Header struct {
	Name     string `dns:"cdomain-name"`
	Rrtype   uint16
	Class    uint16
	Ttl      uint32
	Rdlength uint16 // Length of data after header.
}

// Header returns itself. This is here to make RR_Header implements the RR interface.
func (h *RR_Header) Header() *RR_Header { return h }

// Just to implement the RR interface.
func (h *RR_Header) copy() RR { return nil }

func (h *RR_Header) copyHeader() *RR_Header {
	r := new(RR_Header)
	r.Name = h.Name
	r.Rrtype = h.Rrtype
	r.Class = h.Class
	r.Ttl = h.Ttl
	r.Rdlength = h.Rdlength
	return r
}

func (h *RR_Header) String() string {
	var s string

	if h.Rrtype == TypeOPT {
		s = ";"
		// and maybe other things
	}

	s += sprintName(h.Name) + "\t"
	s += strconv.FormatInt(int64(h.Ttl), 10) + "\t"
	s += Class(h.Class).String() + "\t"
	s += Type(h.Rrtype).String() + "\t"
	return s
}

func (h *RR_Header) len() int {
	l := len(h.Name) + 1
	l += 10 // rrtype(2) + class(2) + ttl(4) + rdlength(2)
	return l
}

// ToRFC3597 converts a known RR to the unknown RR representation from RFC 3597.
func (rr *RFC3597) ToRFC3597(r RR) error {
	buf := make([]byte, r.len()*2)
	off, err := PackRR(r, buf, 0, nil, false)
	if err != nil {
		return err
	}
	buf = buf[:off]
	if int(r.Header().Rdlength) > off {
		return ErrBuf
	}

	rfc3597, _, err := unpackRFC3597(*r.Header(), buf, off-int(r.Header().Rdlength))
	if err != nil {
		return err
	}
	*rr = *rfc3597.(*RFC3597)
	return nil
}
