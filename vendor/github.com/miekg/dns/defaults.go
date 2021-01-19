package dns

import (
	"errors"
	"net"
	"strconv"
	"strings"
)

const hexDigit = "0123456789abcdef"

// Everything is assumed in ClassINET.

// SetReply creates a reply message from a request message.
func (dns *Msg) SetReply(request *Msg) *Msg {
	dns.Id = request.Id
	dns.Response = true
	dns.Opcode = request.Opcode
	if dns.Opcode == OpcodeQuery {
		dns.RecursionDesired = request.RecursionDesired // Copy rd bit
		dns.CheckingDisabled = request.CheckingDisabled // Copy cd bit
	}
	dns.Rcode = RcodeSuccess
	if len(request.Question) > 0 {
		dns.Question = make([]Question, 1)
		dns.Question[0] = request.Question[0]
	}
	return dns
}

// SetQuestion creates a question message, it sets the Question
// section, generates an Id and sets the RecursionDesired (RD)
// bit to true.
func (dns *Msg) SetQuestion(z string, t uint16) *Msg {
	dns.Id = Id()
	dns.RecursionDesired = true
	dns.Question = make([]Question, 1)
	dns.Question[0] = Question{z, t, ClassINET}
	return dns
}

// SetNotify creates a notify message, it sets the Question
// section, generates an Id and sets the Authoritative (AA)
// bit to true.
func (dns *Msg) SetNotify(z string) *Msg {
	dns.Opcode = OpcodeNotify
	dns.Authoritative = true
	dns.Id = Id()
	dns.Question = make([]Question, 1)
	dns.Question[0] = Question{z, TypeSOA, ClassINET}
	return dns
}

// SetRcode creates an error message suitable for the request.
func (dns *Msg) SetRcode(request *Msg, rcode int) *Msg {
	dns.SetReply(request)
	dns.Rcode = rcode
	return dns
}

// SetRcodeFormatError creates a message with FormError set.
func (dns *Msg) SetRcodeFormatError(request *Msg) *Msg {
	dns.Rcode = RcodeFormatError
	dns.Opcode = OpcodeQuery
	dns.Response = true
	dns.Authoritative = false
	dns.Id = request.Id
	return dns
}

// SetUpdate makes the message a dynamic update message. It
// sets the ZONE section to: z, TypeSOA, ClassINET.
func (dns *Msg) SetUpdate(z string) *Msg {
	dns.Id = Id()
	dns.Response = false
	dns.Opcode = OpcodeUpdate
	dns.Compress = false // BIND9 cannot handle compression
	dns.Question = make([]Question, 1)
	dns.Question[0] = Question{z, TypeSOA, ClassINET}
	return dns
}

// SetIxfr creates message for requesting an IXFR.
func (dns *Msg) SetIxfr(z string, serial uint32, ns, mbox string) *Msg {
	dns.Id = Id()
	dns.Question = make([]Question, 1)
	dns.Ns = make([]RR, 1)
	s := new(SOA)
	s.Hdr = RR_Header{z, TypeSOA, ClassINET, defaultTtl, 0}
	s.Serial = serial
	s.Ns = ns
	s.Mbox = mbox
	dns.Question[0] = Question{z, TypeIXFR, ClassINET}
	dns.Ns[0] = s
	return dns
}

// SetAxfr creates message for requesting an AXFR.
func (dns *Msg) SetAxfr(z string) *Msg {
	dns.Id = Id()
	dns.Question = make([]Question, 1)
	dns.Question[0] = Question{z, TypeAXFR, ClassINET}
	return dns
}

// SetTsig appends a TSIG RR to the message.
// This is only a skeleton TSIG RR that is added as the last RR in the
// additional section. The TSIG is calculated when the message is being send.
func (dns *Msg) SetTsig(z, algo string, fudge uint16, timesigned int64) *Msg {
	t := new(TSIG)
	t.Hdr = RR_Header{z, TypeTSIG, ClassANY, 0, 0}
	t.Algorithm = algo
	t.Fudge = fudge
	t.TimeSigned = uint64(timesigned)
	t.OrigId = dns.Id
	dns.Extra = append(dns.Extra, t)
	return dns
}

// SetEdns0 appends a EDNS0 OPT RR to the message.
// TSIG should always the last RR in a message.
func (dns *Msg) SetEdns0(udpsize uint16, do bool) *Msg {
	e := new(OPT)
	e.Hdr.Name = "."
	e.Hdr.Rrtype = TypeOPT
	e.SetUDPSize(udpsize)
	if do {
		e.SetDo()
	}
	dns.Extra = append(dns.Extra, e)
	return dns
}

// IsTsig checks if the message has a TSIG record as the last record
// in the additional section. It returns the TSIG record found or nil.
func (dns *Msg) IsTsig() *TSIG {
	if len(dns.Extra) > 0 {
		if dns.Extra[len(dns.Extra)-1].Header().Rrtype == TypeTSIG {
			return dns.Extra[len(dns.Extra)-1].(*TSIG)
		}
	}
	return nil
}

// IsEdns0 checks if the message has a EDNS0 (OPT) record, any EDNS0
// record in the additional section will do. It returns the OPT record
// found or nil.
func (dns *Msg) IsEdns0() *OPT {
	// RFC 6891, Section 6.1.1 allows the OPT record to appear
	// anywhere in the additional record section, but it's usually at
	// the end so start there.
	for i := len(dns.Extra) - 1; i >= 0; i-- {
		if dns.Extra[i].Header().Rrtype == TypeOPT {
			return dns.Extra[i].(*OPT)
		}
	}
	return nil
}

// popEdns0 is like IsEdns0, but it removes the record from the message.
func (dns *Msg) popEdns0() *OPT {
	// RFC 6891, Section 6.1.1 allows the OPT record to appear
	// anywhere in the additional record section, but it's usually at
	// the end so start there.
	for i := len(dns.Extra) - 1; i >= 0; i-- {
		if dns.Extra[i].Header().Rrtype == TypeOPT {
			opt := dns.Extra[i].(*OPT)
			dns.Extra = append(dns.Extra[:i], dns.Extra[i+1:]...)
			return opt
		}
	}
	return nil
}

// IsDomainName checks if s is a valid domain name, it returns the number of
// labels and true, when a domain name is valid.  Note that non fully qualified
// domain name is considered valid, in this case the last label is counted in
// the number of labels.  When false is returned the number of labels is not
// defined.  Also note that this function is extremely liberal; almost any
// string is a valid domain name as the DNS is 8 bit protocol. It checks if each
// label fits in 63 characters and that the entire name will fit into the 255
// octet wire format limit.
func IsDomainName(s string) (labels int, ok bool) {
	// XXX: The logic in this function was copied from packDomainName and
	// should be kept in sync with that function.

	const lenmsg = 256

	if len(s) == 0 { // Ok, for instance when dealing with update RR without any rdata.
		return 0, false
	}

	s = Fqdn(s)

	// Each dot ends a segment of the name. Except for escaped dots (\.), which
	// are normal dots.

	var (
		off    int
		begin  int
		wasDot bool
	)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\\':
			if off+1 > lenmsg {
				return labels, false
			}

			// check for \DDD
			if i+3 < len(s) && isDigit(s[i+1]) && isDigit(s[i+2]) && isDigit(s[i+3]) {
				i += 3
				begin += 3
			} else {
				i++
				begin++
			}

			wasDot = false
		case '.':
			if wasDot {
				// two dots back to back is not legal
				return labels, false
			}
			wasDot = true

			labelLen := i - begin
			if labelLen >= 1<<6 { // top two bits of length must be clear
				return labels, false
			}

			// off can already (we're in a loop) be bigger than lenmsg
			// this happens when a name isn't fully qualified
			off += 1 + labelLen
			if off > lenmsg {
				return labels, false
			}

			labels++
			begin = i + 1
		default:
			wasDot = false
		}
	}

	return labels, true
}

// IsSubDomain checks if child is indeed a child of the parent. If child and parent
// are the same domain true is returned as well.
func IsSubDomain(parent, child string) bool {
	// Entire child is contained in parent
	return CompareDomainName(parent, child) == CountLabel(parent)
}

// IsMsg sanity checks buf and returns an error if it isn't a valid DNS packet.
// The checking is performed on the binary payload.
func IsMsg(buf []byte) error {
	// Header
	if len(buf) < headerSize {
		return errors.New("dns: bad message header")
	}
	// Header: Opcode
	// TODO(miek): more checks here, e.g. check all header bits.
	return nil
}

// IsFqdn checks if a domain name is fully qualified.
func IsFqdn(s string) bool {
	s2 := strings.TrimSuffix(s, ".")
	if s == s2 {
		return false
	}

	i := strings.LastIndexFunc(s2, func(r rune) bool {
		return r != '\\'
	})

	// Test whether we have an even number of escape sequences before
	// the dot or none.
	return (len(s2)-i)%2 != 0
}

// IsRRset checks if a set of RRs is a valid RRset as defined by RFC 2181.
// This means the RRs need to have the same type, name, and class. Returns true
// if the RR set is valid, otherwise false.
func IsRRset(rrset []RR) bool {
	if len(rrset) == 0 {
		return false
	}
	if len(rrset) == 1 {
		return true
	}
	rrHeader := rrset[0].Header()
	rrType := rrHeader.Rrtype
	rrClass := rrHeader.Class
	rrName := rrHeader.Name

	for _, rr := range rrset[1:] {
		curRRHeader := rr.Header()
		if curRRHeader.Rrtype != rrType || curRRHeader.Class != rrClass || curRRHeader.Name != rrName {
			// Mismatch between the records, so this is not a valid rrset for
			//signing/verifying
			return false
		}
	}

	return true
}

// Fqdn return the fully qualified domain name from s.
// If s is already fully qualified, it behaves as the identity function.
func Fqdn(s string) string {
	if IsFqdn(s) {
		return s
	}
	return s + "."
}

// CanonicalName returns the domain name in canonical form. A name in canonical
// form is lowercase and fully qualified. See Section 6.2 in RFC 4034.
func CanonicalName(s string) string {
	return strings.ToLower(Fqdn(s))
}

// Copied from the official Go code.

// ReverseAddr returns the in-addr.arpa. or ip6.arpa. hostname of the IP
// address suitable for reverse DNS (PTR) record lookups or an error if it fails
// to parse the IP address.
func ReverseAddr(addr string) (arpa string, err error) {
	ip := net.ParseIP(addr)
	if ip == nil {
		return "", &Error{err: "unrecognized address: " + addr}
	}
	if v4 := ip.To4(); v4 != nil {
		buf := make([]byte, 0, net.IPv4len*4+len("in-addr.arpa."))
		// Add it, in reverse, to the buffer
		for i := len(v4) - 1; i >= 0; i-- {
			buf = strconv.AppendInt(buf, int64(v4[i]), 10)
			buf = append(buf, '.')
		}
		// Append "in-addr.arpa." and return (buf already has the final .)
		buf = append(buf, "in-addr.arpa."...)
		return string(buf), nil
	}
	// Must be IPv6
	buf := make([]byte, 0, net.IPv6len*4+len("ip6.arpa."))
	// Add it, in reverse, to the buffer
	for i := len(ip) - 1; i >= 0; i-- {
		v := ip[i]
		buf = append(buf, hexDigit[v&0xF])
		buf = append(buf, '.')
		buf = append(buf, hexDigit[v>>4])
		buf = append(buf, '.')
	}
	// Append "ip6.arpa." and return (buf already has the final .)
	buf = append(buf, "ip6.arpa."...)
	return string(buf), nil
}

// String returns the string representation for the type t.
func (t Type) String() string {
	if t1, ok := TypeToString[uint16(t)]; ok {
		return t1
	}
	return "TYPE" + strconv.Itoa(int(t))
}

// String returns the string representation for the class c.
func (c Class) String() string {
	if s, ok := ClassToString[uint16(c)]; ok {
		// Only emit mnemonics when they are unambiguous, specially ANY is in both.
		if _, ok := StringToType[s]; !ok {
			return s
		}
	}
	return "CLASS" + strconv.Itoa(int(c))
}

// String returns the string representation for the name n.
func (n Name) String() string {
	return sprintName(string(n))
}
