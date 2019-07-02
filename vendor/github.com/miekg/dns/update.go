package dns

// NameUsed sets the RRs in the prereq section to
// "Name is in use" RRs. RFC 2136 section 2.4.4.
func (u *Msg) NameUsed(rr []RR) {
	if u.Answer == nil {
		u.Answer = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		u.Answer = append(u.Answer, &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassANY}})
	}
}

// NameNotUsed sets the RRs in the prereq section to
// "Name is in not use" RRs. RFC 2136 section 2.4.5.
func (u *Msg) NameNotUsed(rr []RR) {
	if u.Answer == nil {
		u.Answer = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		u.Answer = append(u.Answer, &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassNONE}})
	}
}

// Used sets the RRs in the prereq section to
// "RRset exists (value dependent -- with rdata)" RRs. RFC 2136 section 2.4.2.
func (u *Msg) Used(rr []RR) {
	if len(u.Question) == 0 {
		panic("dns: empty question section")
	}
	if u.Answer == nil {
		u.Answer = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		r.Header().Class = u.Question[0].Qclass
		u.Answer = append(u.Answer, r)
	}
}

// RRsetUsed sets the RRs in the prereq section to
// "RRset exists (value independent -- no rdata)" RRs. RFC 2136 section 2.4.1.
func (u *Msg) RRsetUsed(rr []RR) {
	if u.Answer == nil {
		u.Answer = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		h := r.Header()
		u.Answer = append(u.Answer, &ANY{Hdr: RR_Header{Name: h.Name, Ttl: 0, Rrtype: h.Rrtype, Class: ClassANY}})
	}
}

// RRsetNotUsed sets the RRs in the prereq section to
// "RRset does not exist" RRs. RFC 2136 section 2.4.3.
func (u *Msg) RRsetNotUsed(rr []RR) {
	if u.Answer == nil {
		u.Answer = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		h := r.Header()
		u.Answer = append(u.Answer, &ANY{Hdr: RR_Header{Name: h.Name, Ttl: 0, Rrtype: h.Rrtype, Class: ClassNONE}})
	}
}

// Insert creates a dynamic update packet that adds an complete RRset, see RFC 2136 section 2.5.1.
func (u *Msg) Insert(rr []RR) {
	if len(u.Question) == 0 {
		panic("dns: empty question section")
	}
	if u.Ns == nil {
		u.Ns = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		r.Header().Class = u.Question[0].Qclass
		u.Ns = append(u.Ns, r)
	}
}

// RemoveRRset creates a dynamic update packet that deletes an RRset, see RFC 2136 section 2.5.2.
func (u *Msg) RemoveRRset(rr []RR) {
	if u.Ns == nil {
		u.Ns = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		h := r.Header()
		u.Ns = append(u.Ns, &ANY{Hdr: RR_Header{Name: h.Name, Ttl: 0, Rrtype: h.Rrtype, Class: ClassANY}})
	}
}

// RemoveName creates a dynamic update packet that deletes all RRsets of a name, see RFC 2136 section 2.5.3
func (u *Msg) RemoveName(rr []RR) {
	if u.Ns == nil {
		u.Ns = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		u.Ns = append(u.Ns, &ANY{Hdr: RR_Header{Name: r.Header().Name, Ttl: 0, Rrtype: TypeANY, Class: ClassANY}})
	}
}

// Remove creates a dynamic update packet deletes RR from a RRSset, see RFC 2136 section 2.5.4
func (u *Msg) Remove(rr []RR) {
	if u.Ns == nil {
		u.Ns = make([]RR, 0, len(rr))
	}
	for _, r := range rr {
		h := r.Header()
		h.Class = ClassNONE
		h.Ttl = 0
		u.Ns = append(u.Ns, r)
	}
}
