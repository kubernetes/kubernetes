// Package dns implements a full featured interface to the Domain Name System.
// Server- and client-side programming is supported.
// The package allows complete control over what is send out to the DNS. The package
// API follows the less-is-more principle, by presenting a small, clean interface.
//
// The package dns supports (asynchronous) querying/replying, incoming/outgoing zone transfers,
// TSIG, EDNS0, dynamic updates, notifies and DNSSEC validation/signing.
// Note that domain names MUST be fully qualified, before sending them, unqualified
// names in a message will result in a packing failure.
//
// Resource records are native types. They are not stored in wire format.
// Basic usage pattern for creating a new resource record:
//
//      r := new(dns.MX)
//      r.Hdr = dns.RR_Header{Name: "miek.nl.", Rrtype: dns.TypeMX, Class: dns.ClassINET, Ttl: 3600}
//      r.Preference = 10
//      r.Mx = "mx.miek.nl."
//
// Or directly from a string:
//
//      mx, err := dns.NewRR("miek.nl. 3600 IN MX 10 mx.miek.nl.")
//
// Or when the default TTL (3600) and class (IN) suit you:
//
//      mx, err := dns.NewRR("miek.nl. MX 10 mx.miek.nl.")
//
// Or even:
//
//      mx, err := dns.NewRR("$ORIGIN nl.\nmiek 1H IN MX 10 mx.miek")
//
// In the DNS messages are exchanged, these messages contain resource
// records (sets).  Use pattern for creating a message:
//
//      m := new(dns.Msg)
//      m.SetQuestion("miek.nl.", dns.TypeMX)
//
// Or when not certain if the domain name is fully qualified:
//
//	m.SetQuestion(dns.Fqdn("miek.nl"), dns.TypeMX)
//
// The message m is now a message with the question section set to ask
// the MX records for the miek.nl. zone.
//
// The following is slightly more verbose, but more flexible:
//
//      m1 := new(dns.Msg)
//      m1.Id = dns.Id()
//      m1.RecursionDesired = true
//      m1.Question = make([]dns.Question, 1)
//      m1.Question[0] = dns.Question{"miek.nl.", dns.TypeMX, dns.ClassINET}
//
// After creating a message it can be send.
// Basic use pattern for synchronous querying the DNS at a
// server configured on 127.0.0.1 and port 53:
//
//      c := new(dns.Client)
//      in, rtt, err := c.Exchange(m1, "127.0.0.1:53")
//
// Suppressing
// multiple outstanding queries (with the same question, type and class) is as easy as setting:
//
//	c.SingleInflight = true
//
// If these "advanced" features are not needed, a simple UDP query can be send,
// with:
//
//	in, err := dns.Exchange(m1, "127.0.0.1:53")
//
// When this functions returns you will get dns message. A dns message consists
// out of four sections.
// The question section: in.Question, the answer section: in.Answer,
// the authority section: in.Ns and the additional section: in.Extra.
//
// Each of these sections (except the Question section) contain a []RR. Basic
// use pattern for accessing the rdata of a TXT RR as the first RR in
// the Answer section:
//
//	if t, ok := in.Answer[0].(*dns.TXT); ok {
//		// do something with t.Txt
//	}
//
// Domain Name and TXT Character String Representations
//
// Both domain names and TXT character strings are converted to presentation
// form both when unpacked and when converted to strings.
//
// For TXT character strings, tabs, carriage returns and line feeds will be
// converted to \t, \r and \n respectively. Back slashes and quotations marks
// will be escaped. Bytes below 32 and above 127 will be converted to \DDD
// form.
//
// For domain names, in addition to the above rules brackets, periods,
// spaces, semicolons and the at symbol are escaped.
package dns

import (
	"strconv"
)

const (
	year68         = 1 << 31 // For RFC1982 (Serial Arithmetic) calculations in 32 bits.
	DefaultMsgSize = 4096    // Standard default for larger than 512 bytes.
	MinMsgSize     = 512     // Minimal size of a DNS packet.
	MaxMsgSize     = 65536   // Largest possible DNS packet.
	defaultTtl     = 3600    // Default TTL.
)

// Error represents a DNS error
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
	// len returns the length (in octects) of the uncompressed RR in wire format.
	len() int
}

// DNS resource records.
// There are many types of RRs,
// but they all share the same header.
type RR_Header struct {
	Name     string `dns:"cdomain-name"`
	Rrtype   uint16
	Class    uint16
	Ttl      uint32
	Rdlength uint16 // length of data after header
}

func (h *RR_Header) Header() *RR_Header { return h }

// Just to imlement the RR interface
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

// ToRFC3597 converts a known RR to the unknown RR representation
// from RFC 3597.
func (rr *RFC3597) ToRFC3597(r RR) error {
	buf := make([]byte, r.len()*2)
	off, err := PackStruct(r, buf, 0)
	if err != nil {
		return err
	}
	buf = buf[:off]
	rawSetRdlength(buf, 0, off)
	_, err = UnpackStruct(rr, buf, 0)
	if err != nil {
		return err
	}
	return nil
}
