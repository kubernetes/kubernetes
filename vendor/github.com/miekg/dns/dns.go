package dns

import "strconv"

const (
	year68     = 1 << 31 // For RFC1982 (Serial Arithmetic) calculations in 32 bits.
	defaultTtl = 3600    // Default internal TTL.

	// DefaultMsgSize is the standard default for messages larger than 512 bytes.
	DefaultMsgSize = 4096
	// MinMsgSize is the minimal size of a DNS packet.
	MinMsgSize = 512
	// MaxMsgSize is the largest possible DNS packet.
	MaxMsgSize = 65535
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

	// len returns the length (in octets) of the compressed or uncompressed RR in wire format.
	//
	// If compression is nil, the uncompressed size will be returned, otherwise the compressed
	// size will be returned and domain names will be added to the map for future compression.
	len(off int, compression map[string]struct{}) int

	// pack packs the records RDATA into wire format. The header will
	// already have been packed into msg.
	pack(msg []byte, off int, compression compressionMap, compress bool) (off1 int, err error)

	// unpack unpacks an RR from wire format.
	//
	// This will only be called on a new and empty RR type with only the header populated. It
	// will only be called if the record's RDATA is non-empty.
	unpack(msg []byte, off int) (off1 int, err error)

	// parse parses an RR from zone file format.
	//
	// This will only be called on a new and empty RR type with only the header populated.
	parse(c *zlexer, origin string) *ParseError

	// isDuplicate returns whether the two RRs are duplicates.
	isDuplicate(r2 RR) bool
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

func (h *RR_Header) len(off int, compression map[string]struct{}) int {
	l := domainNameLen(h.Name, off, compression, true)
	l += 10 // rrtype(2) + class(2) + ttl(4) + rdlength(2)
	return l
}

func (h *RR_Header) pack(msg []byte, off int, compression compressionMap, compress bool) (off1 int, err error) {
	// RR_Header has no RDATA to pack.
	return off, nil
}

func (h *RR_Header) unpack(msg []byte, off int) (int, error) {
	panic("dns: internal error: unpack should never be called on RR_Header")
}

func (h *RR_Header) parse(c *zlexer, origin string) *ParseError {
	panic("dns: internal error: parse should never be called on RR_Header")
}

// ToRFC3597 converts a known RR to the unknown RR representation from RFC 3597.
func (rr *RFC3597) ToRFC3597(r RR) error {
	buf := make([]byte, Len(r)*2)
	headerEnd, off, err := packRR(r, buf, 0, compressionMap{}, false)
	if err != nil {
		return err
	}
	buf = buf[:off]

	*rr = RFC3597{Hdr: *r.Header()}
	rr.Hdr.Rdlength = uint16(off - headerEnd)

	if noRdata(rr.Hdr) {
		return nil
	}

	_, err = rr.unpack(buf, headerEnd)
	if err != nil {
		return err
	}

	return nil
}
