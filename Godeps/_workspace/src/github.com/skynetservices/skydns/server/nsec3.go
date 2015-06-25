// Copyright (c) 2013 Erik St. Martin, Brian Ketelsen. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"encoding/base32"
	"strings"

	"github.com/miekg/dns"
)

// Do DNSSEC NXDOMAIN with NSEC3 whitelies: rfc 7129, appendix B.
// The closest encloser will be qname - the left most label and the
// next closer will be the full qname which we then will deny.
// Idem for source of synthesis.

func (s *server) Denial(m *dns.Msg) {
	if m.Rcode == dns.RcodeNameError {
		// ce is qname minus the left label
		idx := dns.Split(m.Question[0].Name)
		ce := m.Question[0].Name[idx[1]:]

		nsec3ce, nsec3wildcard := newNSEC3CEandWildcard(s.config.Domain, ce, s.config.MinTtl)
		// Add ce and wildcard
		m.Ns = append(m.Ns, nsec3ce)
		m.Ns = append(m.Ns, nsec3wildcard)
		// Deny Qname nsec3
		m.Ns = append(m.Ns, s.newNSEC3NameError(m.Question[0].Name))
	}
	if m.Rcode == dns.RcodeSuccess && len(m.Ns) == 1 {
		// NODATA
		if _, ok := m.Ns[0].(*dns.SOA); ok {
			m.Ns = append(m.Ns, s.newNSEC3NoData(m.Question[0].Name))
		}
	}
}

func packBase32(s string) []byte {
	b32len := base32.HexEncoding.DecodedLen(len(s))
	buf := make([]byte, b32len)
	n, _ := base32.HexEncoding.Decode(buf, []byte(s))
	buf = buf[:n]
	return buf
}

func unpackBase32(b []byte) string {
	b32 := make([]byte, base32.HexEncoding.EncodedLen(len(b)))
	base32.HexEncoding.Encode(b32, b)
	return string(b32)
}

// newNSEC3NameError returns the NSEC3 record needed to denial qname.
func (s *server) newNSEC3NameError(qname string) *dns.NSEC3 {
	n := new(dns.NSEC3)
	n.Hdr.Class = dns.ClassINET
	n.Hdr.Rrtype = dns.TypeNSEC3
	n.Hdr.Ttl = s.config.MinTtl
	n.Hash = dns.SHA1
	n.Flags = 0
	n.Salt = ""
	n.TypeBitMap = []uint16{}

	covername := dns.HashName(qname, dns.SHA1, 0, "")

	buf := packBase32(covername)
	byteArith(buf, false) // one before
	n.Hdr.Name = appendDomain(strings.ToLower(unpackBase32(buf)), s.config.Domain)
	byteArith(buf, true) // one next
	byteArith(buf, true) // and another one
	n.NextDomain = unpackBase32(buf)
	return n
}

// newNSEC3NoData returns the NSEC3 record needed to denial the types
func (s *server) newNSEC3NoData(qname string) *dns.NSEC3 {
	n := new(dns.NSEC3)
	n.Hdr.Class = dns.ClassINET
	n.Hdr.Rrtype = dns.TypeNSEC3
	n.Hdr.Ttl = s.config.MinTtl
	n.Hash = dns.SHA1
	n.Flags = 0
	n.Salt = ""
	n.TypeBitMap = []uint16{dns.TypeA, dns.TypeAAAA, dns.TypeSRV, dns.TypeRRSIG}

	n.Hdr.Name = dns.HashName(qname, dns.SHA1, 0, "")
	buf := packBase32(n.Hdr.Name)
	byteArith(buf, true) // one next
	n.NextDomain = unpackBase32(buf)

	n.Hdr.Name += appendDomain("", s.config.Domain)
	return n
}

// newNSEC3CEandWildcard returns the NSEC3 for the closest encloser
// and the NSEC3 that denies that wildcard at that level.
func newNSEC3CEandWildcard(apex, ce string, ttl uint32) (*dns.NSEC3, *dns.NSEC3) {
	n1 := new(dns.NSEC3)
	n1.Hdr.Class = dns.ClassINET
	n1.Hdr.Rrtype = dns.TypeNSEC3
	n1.Hdr.Ttl = ttl
	n1.Hash = dns.SHA1
	n1.Flags = 0
	n1.Iterations = 0
	n1.Salt = ""
	// for the apex we need another bitmap
	n1.TypeBitMap = []uint16{dns.TypeA, dns.TypeAAAA, dns.TypeSRV, dns.TypeRRSIG}
	prev := dns.HashName(ce, dns.SHA1, n1.Iterations, n1.Salt)
	n1.Hdr.Name = strings.ToLower(prev) + "." + apex
	buf := packBase32(prev)
	byteArith(buf, true) // one next
	n1.NextDomain = unpackBase32(buf)

	n2 := new(dns.NSEC3)
	n2.Hdr.Class = dns.ClassINET
	n2.Hdr.Rrtype = dns.TypeNSEC3
	n2.Hdr.Ttl = ttl
	n2.Hash = dns.SHA1
	n2.Flags = 0
	n2.Iterations = 0
	n2.Salt = ""

	prev = dns.HashName("*."+ce, dns.SHA1, n2.Iterations, n2.Salt)
	buf = packBase32(prev)
	byteArith(buf, false) // one before
	n2.Hdr.Name = appendDomain(strings.ToLower(unpackBase32(buf)), apex)
	byteArith(buf, true) // one next
	byteArith(buf, true) // and another one
	n2.NextDomain = unpackBase32(buf)

	return n1, n2
}

// byteArith adds either 1 or -1 to b, there is no check for under- or overflow.
func byteArith(b []byte, x bool) {
	if x {
		for i := len(b) - 1; i >= 0; i-- {
			if b[i] == 255 {
				b[i] = 0
				continue
			}
			b[i]++
			return
		}
	}
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] == 0 {
			b[i] = 255
			continue
		}
		b[i]--
		return
	}
}
