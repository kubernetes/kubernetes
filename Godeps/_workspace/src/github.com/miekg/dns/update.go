// DYNAMIC UPDATES
//
// Dynamic updates reuses the DNS message format, but renames three of
// the sections. Question is Zone, Answer is Prerequisite, Authority is
// Update, only the Additional is not renamed. See RFC 2136 for the gory details.
//
// You can set a rather complex set of rules for the existence of absence of
// certain resource records or names in a zone to specify if resource records
// should be added or removed. The table from RFC 2136 supplemented with the Go
// DNS function shows which functions exist to specify the prerequisites.
//
// 3.2.4 - Table Of Metavalues Used In Prerequisite Section
//
//   CLASS    TYPE     RDATA    Meaning                    Function
//   --------------------------------------------------------------
//   ANY      ANY      empty    Name is in use             dns.NameUsed
//   ANY      rrset    empty    RRset exists (value indep) dns.RRsetUsed
//   NONE     ANY      empty    Name is not in use         dns.NameNotUsed
//   NONE     rrset    empty    RRset does not exist       dns.RRsetNotUsed
//   zone     rrset    rr       RRset exists (value dep)   dns.Used
//
// The prerequisite section can also be left empty.
// If you have decided on the prerequisites you can tell what RRs should
// be added or deleted. The next table shows the options you have and
// what functions to call.
//
// 3.4.2.6 - Table Of Metavalues Used In Update Section
//
//   CLASS    TYPE     RDATA    Meaning                     Function
//   ---------------------------------------------------------------
//   ANY      ANY      empty    Delete all RRsets from name dns.RemoveName
//   ANY      rrset    empty    Delete an RRset             dns.RemoveRRset
//   NONE     rrset    rr       Delete an RR from RRset     dns.Remove
//   zone     rrset    rr       Add to an RRset             dns.Insert
//
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
	m := make(map[RR_Header]struct{})
	u.Ns = make([]RR, 0, len(rr))
	for _, r := range rr {
		h := *r.Header().copyHeader()
		h.Class = ClassANY
		h.Ttl = 0
		h.Rdlength = 0
		if _, ok := m[h]; ok {
			continue
		}
		m[h] = struct{}{}
		u.Ns = append(u.Ns, &ANY{h})
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
