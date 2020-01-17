package dns

//go:generate go run duplicate_generate.go

// IsDuplicate checks of r1 and r2 are duplicates of each other, excluding the TTL.
// So this means the header data is equal *and* the RDATA is the same. Return true
// is so, otherwise false.
// It's is a protocol violation to have identical RRs in a message.
func IsDuplicate(r1, r2 RR) bool {
	// Check whether the record header is identical.
	if !r1.Header().isDuplicate(r2.Header()) {
		return false
	}

	// Check whether the RDATA is identical.
	return r1.isDuplicate(r2)
}

func (r1 *RR_Header) isDuplicate(_r2 RR) bool {
	r2, ok := _r2.(*RR_Header)
	if !ok {
		return false
	}
	if r1.Class != r2.Class {
		return false
	}
	if r1.Rrtype != r2.Rrtype {
		return false
	}
	if !isDulicateName(r1.Name, r2.Name) {
		return false
	}
	// ignore TTL
	return true
}

// isDulicateName checks if the domain names s1 and s2 are equal.
func isDulicateName(s1, s2 string) bool { return equal(s1, s2) }
