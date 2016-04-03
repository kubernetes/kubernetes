package dns

import (
	"testing"
	"time"
)

func TestSIG0(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	m := new(Msg)
	m.SetQuestion("example.org.", TypeSOA)
	for _, alg := range []uint8{DSA, ECDSAP256SHA256, ECDSAP384SHA384, RSASHA1, RSASHA256, RSASHA512} {
		algstr := AlgorithmToString[alg]
		keyrr := new(KEY)
		keyrr.Hdr.Name = algstr + "."
		keyrr.Hdr.Rrtype = TypeKEY
		keyrr.Hdr.Class = ClassINET
		keyrr.Algorithm = alg
		keysize := 1024
		switch alg {
		case ECDSAP256SHA256:
			keysize = 256
		case ECDSAP384SHA384:
			keysize = 384
		}
		pk, err := keyrr.Generate(keysize)
		if err != nil {
			t.Logf("Failed to generate key for “%s”: %v", algstr, err)
			t.Fail()
			continue
		}
		now := uint32(time.Now().Unix())
		sigrr := new(SIG)
		sigrr.Hdr.Name = "."
		sigrr.Hdr.Rrtype = TypeSIG
		sigrr.Hdr.Class = ClassANY
		sigrr.Algorithm = alg
		sigrr.Expiration = now + 300
		sigrr.Inception = now - 300
		sigrr.KeyTag = keyrr.KeyTag()
		sigrr.SignerName = keyrr.Hdr.Name
		mb, err := sigrr.Sign(pk, m)
		if err != nil {
			t.Logf("Failed to sign message using “%s”: %v", algstr, err)
			t.Fail()
			continue
		}
		m := new(Msg)
		if err := m.Unpack(mb); err != nil {
			t.Logf("Failed to unpack message signed using “%s”: %v", algstr, err)
			t.Fail()
			continue
		}
		if len(m.Extra) != 1 {
			t.Logf("Missing SIG for message signed using “%s”", algstr)
			t.Fail()
			continue
		}
		var sigrrwire *SIG
		switch rr := m.Extra[0].(type) {
		case *SIG:
			sigrrwire = rr
		default:
			t.Logf("Expected SIG RR, instead: %v", rr)
			t.Fail()
			continue
		}
		for _, rr := range []*SIG{sigrr, sigrrwire} {
			id := "sigrr"
			if rr == sigrrwire {
				id = "sigrrwire"
			}
			if err := rr.Verify(keyrr, mb); err != nil {
				t.Logf("Failed to verify “%s” signed SIG(%s): %v", algstr, id, err)
				t.Fail()
				continue
			}
		}
		mb[13]++
		if err := sigrr.Verify(keyrr, mb); err == nil {
			t.Logf("Verify succeeded on an altered message using “%s”", algstr)
			t.Fail()
			continue
		}
		sigrr.Expiration = 2
		sigrr.Inception = 1
		mb, _ = sigrr.Sign(pk, m)
		if err := sigrr.Verify(keyrr, mb); err == nil {
			t.Logf("Verify succeeded on an expired message using “%s”", algstr)
			t.Fail()
			continue
		}
	}
}
