// Package msg defines the Service structure which is used for service discovery.
package msg

import (
	"fmt"
	"net"
	"strings"

	"github.com/miekg/dns"
)

// Service defines a discoverable service in etcd. It is the rdata from a SRV
// record, but with a twist.  Host (Target in SRV) must be a domain name, but
// if it looks like an IP address (4/6), we will treat it like an IP address.
type Service struct {
	Host     string `json:"host,omitempty"`
	Port     int    `json:"port,omitempty"`
	Priority int    `json:"priority,omitempty"`
	Weight   int    `json:"weight,omitempty"`
	Text     string `json:"text,omitempty"`
	Mail     bool   `json:"mail,omitempty"` // Be an MX record. Priority becomes Preference.
	TTL      uint32 `json:"ttl,omitempty"`

	// When a SRV record with a "Host: IP-address" is added, we synthesize
	// a srv.Target domain name.  Normally we convert the full Key where
	// the record lives to a DNS name and use this as the srv.Target.  When
	// TargetStrip > 0 we strip the left most TargetStrip labels from the
	// DNS name.
	TargetStrip int `json:"targetstrip,omitempty"`

	// Group is used to group (or *not* to group) different services
	// together. Services with an identical Group are returned in the same
	// answer.
	Group string `json:"group,omitempty"`

	// Etcd key where we found this service and ignored from json un-/marshalling
	Key string `json:"-"`
}

// RR returns an RR representation of s. It is in a condensed form to minimize space
// when this is returned in a DNS message.
// The RR will look like:
//	1.rails.production.east.skydns.local. 300 CH TXT "service1.example.com:8080(10,0,,false)[0,]"
//                      etcd Key              Ttl               Host:Port          <   see below   >
// between parens: (Priority, Weight, Text (only first 200 bytes!), Mail)
// between blockquotes: [TargetStrip,Group]
// If the record is synthesised by CoreDNS (i.e. no lookup in etcd happened):
//
//	TODO(miek): what to put here?
//
func (s *Service) RR() *dns.TXT {
	l := len(s.Text)
	if l > 200 {
		l = 200
	}
	t := new(dns.TXT)
	t.Hdr.Class = dns.ClassCHAOS
	t.Hdr.Ttl = s.TTL
	t.Hdr.Rrtype = dns.TypeTXT
	t.Hdr.Name = Domain(s.Key)

	t.Txt = make([]string, 1)
	t.Txt[0] = fmt.Sprintf("%s:%d(%d,%d,%s,%t)[%d,%s]",
		s.Host, s.Port,
		s.Priority, s.Weight, s.Text[:l], s.Mail,
		s.TargetStrip, s.Group)
	return t
}

// NewSRV returns a new SRV record based on the Service.
func (s *Service) NewSRV(name string, weight uint16) *dns.SRV {
	host := targetStrip(dns.Fqdn(s.Host), s.TargetStrip)

	return &dns.SRV{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeSRV, Class: dns.ClassINET, Ttl: s.TTL},
		Priority: uint16(s.Priority), Weight: weight, Port: uint16(s.Port), Target: dns.Fqdn(host)}
}

// NewMX returns a new MX record based on the Service.
func (s *Service) NewMX(name string) *dns.MX {
	host := targetStrip(dns.Fqdn(s.Host), s.TargetStrip)

	return &dns.MX{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeMX, Class: dns.ClassINET, Ttl: s.TTL},
		Preference: uint16(s.Priority), Mx: host}
}

// NewA returns a new A record based on the Service.
func (s *Service) NewA(name string, ip net.IP) *dns.A {
	return &dns.A{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeA, Class: dns.ClassINET, Ttl: s.TTL}, A: ip}
}

// NewAAAA returns a new AAAA record based on the Service.
func (s *Service) NewAAAA(name string, ip net.IP) *dns.AAAA {
	return &dns.AAAA{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeAAAA, Class: dns.ClassINET, Ttl: s.TTL}, AAAA: ip}
}

// NewCNAME returns a new CNAME record based on the Service.
func (s *Service) NewCNAME(name string, target string) *dns.CNAME {
	return &dns.CNAME{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeCNAME, Class: dns.ClassINET, Ttl: s.TTL}, Target: dns.Fqdn(target)}
}

// NewTXT returns a new TXT record based on the Service.
func (s *Service) NewTXT(name string) *dns.TXT {
	return &dns.TXT{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeTXT, Class: dns.ClassINET, Ttl: s.TTL}, Txt: split255(s.Text)}
}

// NewPTR returns a new PTR record based on the Service.
func (s *Service) NewPTR(name string, target string) *dns.PTR {
	return &dns.PTR{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypePTR, Class: dns.ClassINET, Ttl: s.TTL}, Ptr: dns.Fqdn(target)}
}

// NewNS returns a new NS record based on the Service.
func (s *Service) NewNS(name string) *dns.NS {
	host := targetStrip(dns.Fqdn(s.Host), s.TargetStrip)
	return &dns.NS{Hdr: dns.RR_Header{Name: name, Rrtype: dns.TypeNS, Class: dns.ClassINET, Ttl: s.TTL}, Ns: host}
}

// Group checks the services in sx, it looks for a Group attribute on the shortest
// keys. If there are multiple shortest keys *and* the group attribute disagrees (and
// is not empty), we don't consider it a group.
// If a group is found, only services with *that* group (or no group) will be returned.
func Group(sx []Service) []Service {
	if len(sx) == 0 {
		return sx
	}

	// Shortest key with group attribute sets the group for this set.
	group := sx[0].Group
	slashes := strings.Count(sx[0].Key, "/")
	length := make([]int, len(sx))
	for i, s := range sx {
		x := strings.Count(s.Key, "/")
		length[i] = x
		if x < slashes {
			if s.Group == "" {
				break
			}
			slashes = x
			group = s.Group
		}
	}

	if group == "" {
		return sx
	}

	ret := []Service{} // with slice-tricks in sx we can prolly save this allocation (TODO)

	for i, s := range sx {
		if s.Group == "" {
			ret = append(ret, s)
			continue
		}

		// Disagreement on the same level
		if length[i] == slashes && s.Group != group {
			return sx
		}

		if s.Group == group {
			ret = append(ret, s)
		}
	}
	return ret
}

// Split255 splits a string into 255 byte chunks.
func split255(s string) []string {
	if len(s) < 255 {
		return []string{s}
	}
	sx := []string{}
	p, i := 0, 255
	for {
		if i <= len(s) {
			sx = append(sx, s[p:i])
		} else {
			sx = append(sx, s[p:])
			break

		}
		p, i = p+255, i+255
	}

	return sx
}

// targetStrip strips "targetstrip" labels from the left side of the fully qualified name.
func targetStrip(name string, targetStrip int) string {
	if targetStrip == 0 {
		return name
	}

	offset, end := 0, false
	for i := 0; i < targetStrip; i++ {
		offset, end = dns.NextLabel(name, offset)
	}
	if end {
		// We overshot the name, use the orignal one.
		offset = 0
	}
	name = name[offset:]
	return name
}
