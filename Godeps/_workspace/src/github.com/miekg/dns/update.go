package dns

// NameUsed sets the RRs in the prereq section to
// "Name is in use" RRs. RFC 2136 section 2.4.4.
func (u *Msg) NameUsed(rr []RR) {
	u.Answer = make([]RR, len(rr))
	for i, r := range rr {
		u.Answer[i] = &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassANY}}
	}
}

// NameNotUsed sets the RRs in the prereq section to
// "Name is in not use" RRs. RFC 2136 section 2.4.5.
func (u *Msg) NameNotUsed(rr []RR) {
	u.Answer = make([]RR, len(rr))
	for i, r := range rr {
		u.Answer[i] = &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassNONE}}
	}
}

// Used sets the RRs in the prereq section to
// "RRset exists (value dependent -- with rdata)" RRs. RFC 2136 section 2.4.2.
func (u *Msg) Used(rr []RR) {
	if len(u.Question) == 0 {
		panic("dns: empty question section")
	}
	u.Answer = make([]RR, len(rr))
	for i, r := range rr {
		u.Answer[i] = r
		u.Answer[i].Header().Class = u.Question[0].Qclass
	}
}

// RRsetUsed sets the RRs in the prereq section to
// "RRset exists (value independent -- no rdata)" RRs. RFC 2136 section 2.4.1.
func (u *Msg) RRsetUsed(rr []RR) {
	u.Answer = make([]RR, len(rr))
	for i, r := range rr {
		u.Answer[i] = r
		u.Answer[i].Header().Class = ClassANY
		u.Answer[i].Header().Ttl = 0
		u.Answer[i].Header().Rdlength = 0
	}
}

// RRsetNotUsed sets the RRs in the prereq section to
// "RRset does not exist" RRs. RFC 2136 section 2.4.3.
func (u *Msg) RRsetNotUsed(rr []RR) {
	u.Answer = make([]RR, len(rr))
	for i, r := range rr {
		u.Answer[i] = r
		u.Answer[i].Header().Class = ClassNONE
		u.Answer[i].Header().Rdlength = 0
		u.Answer[i].Header().Ttl = 0
	}
}

// Insert creates a dynamic update packet that adds an complete RRset, see RFC 2136 section 2.5.1.
func (u *Msg) Insert(rr []RR) {
	if len(u.Question) == 0 {
		panic("dns: empty question section")
	}
	u.Ns = make([]RR, len(rr))
	for i, r := range rr {
		u.Ns[i] = r
		u.Ns[i].Header().Class = u.Question[0].Qclass
	}
}

// RemoveRRset creates a dynamic update packet that deletes an RRset, see RFC 2136 section 2.5.2.
func (u *Msg) RemoveRRset(rr []RR) {
	u.Ns = make([]RR, len(rr))
	for i, r := range rr {
		u.Ns[i] = &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: r.Header().Rrtype, Class: ClassANY}}
	}
}

// RemoveName creates a dynamic update packet that deletes all RRsets of a name, see RFC 2136 section 2.5.3
func (u *Msg) RemoveName(rr []RR) {
	u.Ns = make([]RR, len(rr))
	for i, r := range rr {
		u.Ns[i] = &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassANY}}
	}
}

// Remove creates a dynamic update packet deletes RR from the RRSset, see RFC 2136 section 2.5.4
func (u *Msg) Remove(rr []RR) {
	u.Ns = make([]RR, len(rr))
	for i, r := range rr {
		u.Ns[i] = r
		u.Ns[i].Header().Class = ClassNONE
		u.Ns[i].Header().Ttl = 0
	}
}
