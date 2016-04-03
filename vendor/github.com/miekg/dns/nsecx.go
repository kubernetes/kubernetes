package dns

import (
	"crypto/sha1"
	"hash"
	"io"
	"strings"
)

type saltWireFmt struct {
	Salt string `dns:"size-hex"`
}

// HashName hashes a string (label) according to RFC 5155. It returns the hashed string in
// uppercase.
func HashName(label string, ha uint8, iter uint16, salt string) string {
	saltwire := new(saltWireFmt)
	saltwire.Salt = salt
	wire := make([]byte, DefaultMsgSize)
	n, err := PackStruct(saltwire, wire, 0)
	if err != nil {
		return ""
	}
	wire = wire[:n]
	name := make([]byte, 255)
	off, err := PackDomainName(strings.ToLower(label), name, 0, nil, false)
	if err != nil {
		return ""
	}
	name = name[:off]
	var s hash.Hash
	switch ha {
	case SHA1:
		s = sha1.New()
	default:
		return ""
	}

	// k = 0
	name = append(name, wire...)
	io.WriteString(s, string(name))
	nsec3 := s.Sum(nil)
	// k > 0
	for k := uint16(0); k < iter; k++ {
		s.Reset()
		nsec3 = append(nsec3, wire...)
		io.WriteString(s, string(nsec3))
		nsec3 = s.Sum(nil)
	}
	return toBase32(nsec3)
}

type Denialer interface {
	// Cover will check if the (unhashed) name is being covered by this NSEC or NSEC3.
	Cover(name string) bool
	// Match will check if the ownername matches the (unhashed) name for this NSEC3 or NSEC3.
	Match(name string) bool
}

// Cover implements the Denialer interface.
func (rr *NSEC) Cover(name string) bool {
	return true
}

// Match implements the Denialer interface.
func (rr *NSEC) Match(name string) bool {
	return true
}

// Cover implements the Denialer interface.
func (rr *NSEC3) Cover(name string) bool {
	// FIXME(miek): check if the zones match
	// FIXME(miek): check if we're not dealing with parent nsec3
	hname := HashName(name, rr.Hash, rr.Iterations, rr.Salt)
	labels := Split(rr.Hdr.Name)
	if len(labels) < 2 {
		return false
	}
	hash := strings.ToUpper(rr.Hdr.Name[labels[0] : labels[1]-1]) // -1 to remove the dot
	if hash == rr.NextDomain {
		return false // empty interval
	}
	if hash > rr.NextDomain { // last name, points to apex
		// hname > hash
		// hname > rr.NextDomain
		// TODO(miek)
	}
	if hname <= hash {
		return false
	}
	if hname >= rr.NextDomain {
		return false
	}
	return true
}

// Match implements the Denialer interface.
func (rr *NSEC3) Match(name string) bool {
	// FIXME(miek): Check if we are in the same zone
	hname := HashName(name, rr.Hash, rr.Iterations, rr.Salt)
	labels := Split(rr.Hdr.Name)
	if len(labels) < 2 {
		return false
	}
	hash := strings.ToUpper(rr.Hdr.Name[labels[0] : labels[1]-1]) // -1 to remove the .
	if hash == hname {
		return true
	}
	return false
}
