package dns

import (
	"bytes"
	"crypto/rsa"
	"encoding/hex"
	"fmt"
	"math/rand"
	"net"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"testing/quick"
	"time"
)

func TestDotInName(t *testing.T) {
	buf := make([]byte, 20)
	PackDomainName("aa\\.bb.nl.", buf, 0, nil, false)
	// index 3 must be a real dot
	if buf[3] != '.' {
		t.Error("dot should be a real dot")
	}

	if buf[6] != 2 {
		t.Error("this must have the value 2")
	}
	dom, _, _ := UnpackDomainName(buf, 0)
	// printing it should yield the backspace again
	if dom != "aa\\.bb.nl." {
		t.Error("dot should have been escaped: ", dom)
	}
}

func TestDotLastInLabel(t *testing.T) {
	sample := "aa\\..au."
	buf := make([]byte, 20)
	_, err := PackDomainName(sample, buf, 0, nil, false)
	if err != nil {
		t.Fatalf("unexpected error packing domain: %v", err)
	}
	dom, _, _ := UnpackDomainName(buf, 0)
	if dom != sample {
		t.Fatalf("unpacked domain `%s' doesn't match packed domain", dom)
	}
}

func TestTooLongDomainName(t *testing.T) {
	l := "aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrsssttt."
	dom := l + l + l + l + l + l + l
	_, err := NewRR(dom + " IN A 127.0.0.1")
	if err == nil {
		t.Error("should be too long")
	} else {
		t.Logf("error is %v", err)
	}
	_, err = NewRR("..com. IN A 127.0.0.1")
	if err == nil {
		t.Error("should fail")
	} else {
		t.Logf("error is %v", err)
	}
}

func TestDomainName(t *testing.T) {
	tests := []string{"r\\.gieben.miek.nl.", "www\\.www.miek.nl.",
		"www.*.miek.nl.", "www.*.miek.nl.",
	}
	dbuff := make([]byte, 40)

	for _, ts := range tests {
		if _, err := PackDomainName(ts, dbuff, 0, nil, false); err != nil {
			t.Error("not a valid domain name")
			continue
		}
		n, _, err := UnpackDomainName(dbuff, 0)
		if err != nil {
			t.Error("failed to unpack packed domain name")
			continue
		}
		if ts != n {
			t.Errorf("must be equal: in: %s, out: %s", ts, n)
		}
	}
}

func TestDomainNameAndTXTEscapes(t *testing.T) {
	tests := []byte{'.', '(', ')', ';', ' ', '@', '"', '\\', '\t', '\r', '\n', 0, 255}
	for _, b := range tests {
		rrbytes := []byte{
			1, b, 0, // owner
			byte(TypeTXT >> 8), byte(TypeTXT),
			byte(ClassINET >> 8), byte(ClassINET),
			0, 0, 0, 1, // TTL
			0, 2, 1, b, // Data
		}
		rr1, _, err := UnpackRR(rrbytes, 0)
		if err != nil {
			panic(err)
		}
		s := rr1.String()
		rr2, err := NewRR(s)
		if err != nil {
			t.Errorf("Error parsing unpacked RR's string: %v", err)
			t.Errorf(" Bytes: %v", rrbytes)
			t.Errorf("String: %v", s)
		}
		repacked := make([]byte, len(rrbytes))
		if _, err := PackRR(rr2, repacked, 0, nil, false); err != nil {
			t.Errorf("error packing parsed RR: %v", err)
			t.Errorf(" original Bytes: %v", rrbytes)
			t.Errorf("unpacked Struct: %v", rr1)
			t.Errorf("  parsed Struct: %v", rr2)
		}
		if !bytes.Equal(repacked, rrbytes) {
			t.Error("packed bytes don't match original bytes")
			t.Errorf(" original bytes: %v", rrbytes)
			t.Errorf("   packed bytes: %v", repacked)
			t.Errorf("unpacked struct: %v", rr1)
			t.Errorf("  parsed struct: %v", rr2)
		}
	}
}

func TestTXTEscapeParsing(t *testing.T) {
	test := [][]string{
		{`";"`, `";"`},
		{`\;`, `";"`},
		{`"\t"`, `"\t"`},
		{`"\r"`, `"\r"`},
		{`"\ "`, `" "`},
		{`"\;"`, `";"`},
		{`"\;\""`, `";\""`},
		{`"\(a\)"`, `"(a)"`},
		{`"\(a)"`, `"(a)"`},
		{`"(a\)"`, `"(a)"`},
		{`"(a)"`, `"(a)"`},
		{`"\048"`, `"0"`},
		{`"\` + "\n" + `"`, `"\n"`},
		{`"\` + "\r" + `"`, `"\r"`},
		{`"\` + "\x11" + `"`, `"\017"`},
		{`"\'"`, `"'"`},
	}
	for _, s := range test {
		rr, err := NewRR(fmt.Sprintf("example.com. IN TXT %v", s[0]))
		if err != nil {
			t.Errorf("could not parse %v TXT: %s", s[0], err)
			continue
		}

		txt := sprintTxt(rr.(*TXT).Txt)
		if txt != s[1] {
			t.Errorf("mismatch after parsing `%v` TXT record: `%v` != `%v`", s[0], txt, s[1])
		}
	}
}

func GenerateDomain(r *rand.Rand, size int) []byte {
	dnLen := size % 70 // artificially limit size so there's less to intrepret if a failure occurs
	var dn []byte
	done := false
	for i := 0; i < dnLen && !done; {
		max := dnLen - i
		if max > 63 {
			max = 63
		}
		lLen := max
		if lLen != 0 {
			lLen = int(r.Int31()) % max
		}
		done = lLen == 0
		if done {
			continue
		}
		l := make([]byte, lLen+1)
		l[0] = byte(lLen)
		for j := 0; j < lLen; j++ {
			l[j+1] = byte(rand.Int31())
		}
		dn = append(dn, l...)
		i += 1 + lLen
	}
	return append(dn, 0)
}

func TestDomainQuick(t *testing.T) {
	r := rand.New(rand.NewSource(0))
	f := func(l int) bool {
		db := GenerateDomain(r, l)
		ds, _, err := UnpackDomainName(db, 0)
		if err != nil {
			panic(err)
		}
		buf := make([]byte, 255)
		off, err := PackDomainName(ds, buf, 0, nil, false)
		if err != nil {
			t.Errorf("error packing domain: %v", err)
			t.Errorf(" bytes: %v", db)
			t.Errorf("string: %v", ds)
			return false
		}
		if !bytes.Equal(db, buf[:off]) {
			t.Errorf("repacked domain doesn't match original:")
			t.Errorf("src bytes: %v", db)
			t.Errorf("   string: %v", ds)
			t.Errorf("out bytes: %v", buf[:off])
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

func GenerateTXT(r *rand.Rand, size int) []byte {
	rdLen := size % 300 // artificially limit size so there's less to intrepret if a failure occurs
	var rd []byte
	for i := 0; i < rdLen; {
		max := rdLen - 1
		if max > 255 {
			max = 255
		}
		sLen := max
		if max != 0 {
			sLen = int(r.Int31()) % max
		}
		s := make([]byte, sLen+1)
		s[0] = byte(sLen)
		for j := 0; j < sLen; j++ {
			s[j+1] = byte(rand.Int31())
		}
		rd = append(rd, s...)
		i += 1 + sLen
	}
	return rd
}

// Ok, 2 things. 1) this test breaks with the new functionality of splitting up larger txt
// chunks into 255 byte pieces. 2) I don't like the random nature of this thing, because I can't
// place the quotes where they need to be.
// So either add some code the places the quotes in just the right spots, make this non random
// or do something else.
// Disabled for now. (miek)
func testTXTRRQuick(t *testing.T) {
	s := rand.NewSource(0)
	r := rand.New(s)
	typeAndClass := []byte{
		byte(TypeTXT >> 8), byte(TypeTXT),
		byte(ClassINET >> 8), byte(ClassINET),
		0, 0, 0, 1, // TTL
	}
	f := func(l int) bool {
		owner := GenerateDomain(r, l)
		rdata := GenerateTXT(r, l)
		rrbytes := make([]byte, 0, len(owner)+2+2+4+2+len(rdata))
		rrbytes = append(rrbytes, owner...)
		rrbytes = append(rrbytes, typeAndClass...)
		rrbytes = append(rrbytes, byte(len(rdata)>>8))
		rrbytes = append(rrbytes, byte(len(rdata)))
		rrbytes = append(rrbytes, rdata...)
		rr, _, err := UnpackRR(rrbytes, 0)
		if err != nil {
			panic(err)
		}
		buf := make([]byte, len(rrbytes)*3)
		off, err := PackRR(rr, buf, 0, nil, false)
		if err != nil {
			t.Errorf("pack Error: %v\nRR: %v", err, rr)
			return false
		}
		buf = buf[:off]
		if !bytes.Equal(buf, rrbytes) {
			t.Errorf("packed bytes don't match original bytes")
			t.Errorf("src bytes: %v", rrbytes)
			t.Errorf("   struct: %v", rr)
			t.Errorf("out bytes: %v", buf)
			return false
		}
		if len(rdata) == 0 {
			// string'ing won't produce any data to parse
			return true
		}
		rrString := rr.String()
		rr2, err := NewRR(rrString)
		if err != nil {
			t.Errorf("error parsing own output: %v", err)
			t.Errorf("struct: %v", rr)
			t.Errorf("string: %v", rrString)
			return false
		}
		if rr2.String() != rrString {
			t.Errorf("parsed rr.String() doesn't match original string")
			t.Errorf("original: %v", rrString)
			t.Errorf("  parsed: %v", rr2.String())
			return false
		}

		buf = make([]byte, len(rrbytes)*3)
		off, err = PackRR(rr2, buf, 0, nil, false)
		if err != nil {
			t.Errorf("error packing parsed rr: %v", err)
			t.Errorf("unpacked Struct: %v", rr)
			t.Errorf("         string: %v", rrString)
			t.Errorf("  parsed Struct: %v", rr2)
			return false
		}
		buf = buf[:off]
		if !bytes.Equal(buf, rrbytes) {
			t.Errorf("parsed packed bytes don't match original bytes")
			t.Errorf("   source bytes: %v", rrbytes)
			t.Errorf("unpacked struct: %v", rr)
			t.Errorf("         string: %v", rrString)
			t.Errorf("  parsed struct: %v", rr2)
			t.Errorf(" repacked bytes: %v", buf)
			return false
		}
		return true
	}
	c := &quick.Config{MaxCountScale: 10}
	if err := quick.Check(f, c); err != nil {
		t.Error(err)
	}
}

func TestParseDirectiveMisc(t *testing.T) {
	tests := map[string]string{
		"$ORIGIN miek.nl.\na IN NS b": "a.miek.nl.\t3600\tIN\tNS\tb.miek.nl.",
		"$TTL 2H\nmiek.nl. IN NS b.":  "miek.nl.\t7200\tIN\tNS\tb.",
		"miek.nl. 1D IN NS b.":        "miek.nl.\t86400\tIN\tNS\tb.",
		`name. IN SOA  a6.nstld.com. hostmaster.nic.name. (
        203362132 ; serial
        5m        ; refresh (5 minutes)
        5m        ; retry (5 minutes)
        2w        ; expire (2 weeks)
        300       ; minimum (5 minutes)
)`: "name.\t3600\tIN\tSOA\ta6.nstld.com. hostmaster.nic.name. 203362132 300 300 1209600 300",
		". 3600000  IN  NS ONE.MY-ROOTS.NET.":        ".\t3600000\tIN\tNS\tONE.MY-ROOTS.NET.",
		"ONE.MY-ROOTS.NET. 3600000 IN A 192.168.1.1": "ONE.MY-ROOTS.NET.\t3600000\tIN\tA\t192.168.1.1",
	}
	for i, o := range tests {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestNSEC(t *testing.T) {
	nsectests := map[string]string{
		"nl. IN NSEC3PARAM 1 0 5 30923C44C6CBBB8F":                                                                                                 "nl.\t3600\tIN\tNSEC3PARAM\t1 0 5 30923C44C6CBBB8F",
		"p2209hipbpnm681knjnu0m1febshlv4e.nl. IN NSEC3 1 1 5 30923C44C6CBBB8F P90DG1KE8QEAN0B01613LHQDG0SOJ0TA NS SOA TXT RRSIG DNSKEY NSEC3PARAM": "p2209hipbpnm681knjnu0m1febshlv4e.nl.\t3600\tIN\tNSEC3\t1 1 5 30923C44C6CBBB8F P90DG1KE8QEAN0B01613LHQDG0SOJ0TA NS SOA TXT RRSIG DNSKEY NSEC3PARAM",
		"localhost.dnssex.nl. IN NSEC www.dnssex.nl. A RRSIG NSEC":                                                                                 "localhost.dnssex.nl.\t3600\tIN\tNSEC\twww.dnssex.nl. A RRSIG NSEC",
		"localhost.dnssex.nl. IN NSEC www.dnssex.nl. A RRSIG NSEC TYPE65534":                                                                       "localhost.dnssex.nl.\t3600\tIN\tNSEC\twww.dnssex.nl. A RRSIG NSEC TYPE65534",
		"localhost.dnssex.nl. IN NSEC www.dnssex.nl. A RRSIG NSec Type65534":                                                                       "localhost.dnssex.nl.\t3600\tIN\tNSEC\twww.dnssex.nl. A RRSIG NSEC TYPE65534",
	}
	for i, o := range nsectests {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseLOC(t *testing.T) {
	lt := map[string]string{
		"SW1A2AA.find.me.uk.	LOC	51 30 12.748 N 00 07 39.611 W 0.00m 0.00m 0.00m 0.00m": "SW1A2AA.find.me.uk.\t3600\tIN\tLOC\t51 30 12.748 N 00 07 39.611 W 0m 0.00m 0.00m 0.00m",
		"SW1A2AA.find.me.uk.	LOC	51 0 0.0 N 00 07 39.611 W 0.00m 0.00m 0.00m 0.00m": "SW1A2AA.find.me.uk.\t3600\tIN\tLOC\t51 00 0.000 N 00 07 39.611 W 0m 0.00m 0.00m 0.00m",
	}
	for i, o := range lt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseDS(t *testing.T) {
	dt := map[string]string{
		"example.net. 3600 IN DS 40692 12 3 22261A8B0E0D799183E35E24E2AD6BB58533CBA7E3B14D659E9CA09B 2071398F": "example.net.\t3600\tIN\tDS\t40692 12 3 22261A8B0E0D799183E35E24E2AD6BB58533CBA7E3B14D659E9CA09B2071398F",
	}
	for i, o := range dt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestQuotes(t *testing.T) {
	tests := map[string]string{
		`t.example.com. IN TXT "a bc"`: "t.example.com.\t3600\tIN\tTXT\t\"a bc\"",
		`t.example.com. IN TXT "a
 bc"`: "t.example.com.\t3600\tIN\tTXT\t\"a\\n bc\"",
		`t.example.com. IN TXT ""`:                                                           "t.example.com.\t3600\tIN\tTXT\t\"\"",
		`t.example.com. IN TXT "a"`:                                                          "t.example.com.\t3600\tIN\tTXT\t\"a\"",
		`t.example.com. IN TXT "aa"`:                                                         "t.example.com.\t3600\tIN\tTXT\t\"aa\"",
		`t.example.com. IN TXT "aaa" ;`:                                                      "t.example.com.\t3600\tIN\tTXT\t\"aaa\"",
		`t.example.com. IN TXT "abc" "DEF"`:                                                  "t.example.com.\t3600\tIN\tTXT\t\"abc\" \"DEF\"",
		`t.example.com. IN TXT "abc" ( "DEF" )`:                                              "t.example.com.\t3600\tIN\tTXT\t\"abc\" \"DEF\"",
		`t.example.com. IN TXT aaa ;`:                                                        "t.example.com.\t3600\tIN\tTXT\t\"aaa \"",
		`t.example.com. IN TXT aaa aaa;`:                                                     "t.example.com.\t3600\tIN\tTXT\t\"aaa aaa\"",
		`t.example.com. IN TXT aaa aaa`:                                                      "t.example.com.\t3600\tIN\tTXT\t\"aaa aaa\"",
		`t.example.com. IN TXT aaa`:                                                          "t.example.com.\t3600\tIN\tTXT\t\"aaa\"",
		"cid.urn.arpa. NAPTR 100 50 \"s\" \"z3950+I2L+I2C\"    \"\" _z3950._tcp.gatech.edu.": "cid.urn.arpa.\t3600\tIN\tNAPTR\t100 50 \"s\" \"z3950+I2L+I2C\" \"\" _z3950._tcp.gatech.edu.",
		"cid.urn.arpa. NAPTR 100 50 \"s\" \"rcds+I2C\"         \"\" _rcds._udp.gatech.edu.":  "cid.urn.arpa.\t3600\tIN\tNAPTR\t100 50 \"s\" \"rcds+I2C\" \"\" _rcds._udp.gatech.edu.",
		"cid.urn.arpa. NAPTR 100 50 \"s\" \"http+I2L+I2C+I2R\" \"\" _http._tcp.gatech.edu.":  "cid.urn.arpa.\t3600\tIN\tNAPTR\t100 50 \"s\" \"http+I2L+I2C+I2R\" \"\" _http._tcp.gatech.edu.",
		"cid.urn.arpa. NAPTR 100 10 \"\" \"\" \"/urn:cid:.+@([^\\.]+\\.)(.*)$/\\2/i\" .":     "cid.urn.arpa.\t3600\tIN\tNAPTR\t100 10 \"\" \"\" \"/urn:cid:.+@([^\\.]+\\.)(.*)$/\\2/i\" .",
	}
	for i, o := range tests {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is\n`%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseClass(t *testing.T) {
	tests := map[string]string{
		"t.example.com. IN A 127.0.0.1": "t.example.com.	3600	IN	A	127.0.0.1",
		"t.example.com. CS A 127.0.0.1": "t.example.com.	3600	CS	A	127.0.0.1",
		"t.example.com. CH A 127.0.0.1": "t.example.com.	3600	CH	A	127.0.0.1",
		// ClassANY can not occur in zone files
		// "t.example.com. ANY A 127.0.0.1": "t.example.com.	3600	ANY	A	127.0.0.1",
		"t.example.com. NONE A 127.0.0.1": "t.example.com.	3600	NONE	A	127.0.0.1",
	}
	for i, o := range tests {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is\n`%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestBrace(t *testing.T) {
	tests := map[string]string{
		"(miek.nl.) 3600 IN A 127.0.1.1":                 "miek.nl.\t3600\tIN\tA\t127.0.1.1",
		"miek.nl. (3600) IN MX (10) elektron.atoom.net.": "miek.nl.\t3600\tIN\tMX\t10 elektron.atoom.net.",
		`miek.nl. IN (
                        3600 A 127.0.0.1)`: "miek.nl.\t3600\tIN\tA\t127.0.0.1",
		"(miek.nl.) (A) (127.0.2.1)":                          "miek.nl.\t3600\tIN\tA\t127.0.2.1",
		"miek.nl A 127.0.3.1":                                 "miek.nl.\t3600\tIN\tA\t127.0.3.1",
		"_ssh._tcp.local. 60 IN (PTR) stora._ssh._tcp.local.": "_ssh._tcp.local.\t60\tIN\tPTR\tstora._ssh._tcp.local.",
		"miek.nl. NS ns.miek.nl":                              "miek.nl.\t3600\tIN\tNS\tns.miek.nl.",
		`(miek.nl.) (
                        (IN)
                        (AAAA)
                        (::1) )`: "miek.nl.\t3600\tIN\tAAAA\t::1",
		`(miek.nl.) (
                        (IN)
                        (AAAA)
                        (::1))`: "miek.nl.\t3600\tIN\tAAAA\t::1",
		"miek.nl. IN AAAA ::2": "miek.nl.\t3600\tIN\tAAAA\t::2",
		`((m)(i)ek.(n)l.) (SOA) (soa.) (soa.) (
                                2009032802 ; serial
                                21600      ; refresh (6 hours)
                                7(2)00       ; retry (2 hours)
                                604()800     ; expire (1 week)
                                3600       ; minimum (1 hour)
                        )`: "miek.nl.\t3600\tIN\tSOA\tsoa. soa. 2009032802 21600 7200 604800 3600",
		"miek\\.nl. IN A 127.0.0.10": "miek\\.nl.\t3600\tIN\tA\t127.0.0.10",
		"miek.nl. IN A 127.0.0.11":   "miek.nl.\t3600\tIN\tA\t127.0.0.11",
		"miek.nl. A 127.0.0.12":      "miek.nl.\t3600\tIN\tA\t127.0.0.12",
		`miek.nl.       86400 IN SOA elektron.atoom.net. miekg.atoom.net. (
                                2009032802 ; serial
                                21600      ; refresh (6 hours)
                                7200       ; retry (2 hours)
                                604800     ; expire (1 week)
                                3600       ; minimum (1 hour)
                        )`: "miek.nl.\t86400\tIN\tSOA\telektron.atoom.net. miekg.atoom.net. 2009032802 21600 7200 604800 3600",
	}
	for i, o := range tests {
		rr, err := NewRR(i)
		if err != nil {
			t.Errorf("failed to parse RR: %v\n\t%s", err, i)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseFailure(t *testing.T) {
	tests := []string{"miek.nl. IN A 327.0.0.1",
		"miek.nl. IN AAAA ::x",
		"miek.nl. IN MX a0 miek.nl.",
		"miek.nl aap IN MX mx.miek.nl.",
		"miek.nl 200 IN mxx 10 mx.miek.nl.",
		"miek.nl. inn MX 10 mx.miek.nl.",
		// "miek.nl. IN CNAME ", // actually valid nowadays, zero size rdata
		"miek.nl. IN CNAME ..",
		"miek.nl. PA MX 10 miek.nl.",
		"miek.nl. ) IN MX 10 miek.nl.",
	}

	for _, s := range tests {
		_, err := NewRR(s)
		if err == nil {
			t.Errorf("should have triggered an error: \"%s\"", s)
		}
	}
}

func TestZoneParsing(t *testing.T) {
	// parse_test.db
	db := `
a.example.com.                IN A 127.0.0.1
8db7._openpgpkey.example.com. IN OPENPGPKEY mQCNAzIG
$ORIGIN a.example.com.
test                          IN A 127.0.0.1
                              IN SSHFP   1 2 (
                                           BC6533CDC95A79078A39A56EA7635984ED655318ADA9
                                           B6159E30723665DA95BB )
$ORIGIN b.example.com.
test                          IN CNAME test.a.example.com.
`
	start := time.Now().UnixNano()
	to := ParseZone(strings.NewReader(db), "", "parse_test.db")
	var i int
	for x := range to {
		i++
		if x.Error != nil {
			t.Error(x.Error)
			continue
		}
		t.Log(x.RR)
	}
	delta := time.Now().UnixNano() - start
	t.Logf("%d RRs parsed in %.2f s (%.2f RR/s)", i, float32(delta)/1e9, float32(i)/(float32(delta)/1e9))
}

func ExampleParseZone() {
	zone := `$ORIGIN .
$TTL 3600       ; 1 hour
name                    IN SOA  a6.nstld.com. hostmaster.nic.name. (
                                203362132  ; serial
                                300        ; refresh (5 minutes)
                                300        ; retry (5 minutes)
                                1209600    ; expire (2 weeks)
                                300        ; minimum (5 minutes)
                                )
$TTL 10800      ; 3 hours
name.	10800	IN	NS	name.
               IN       NS      g6.nstld.com.
               7200     NS      h6.nstld.com.
             3600 IN    NS      j6.nstld.com.
             IN 3600    NS      k6.nstld.com.
                        NS      l6.nstld.com.
                        NS      a6.nstld.com.
                        NS      c6.nstld.com.
                        NS      d6.nstld.com.
                        NS      f6.nstld.com.
                        NS      m6.nstld.com.
(
			NS	m7.nstld.com.
)
$ORIGIN name.
0-0onlus                NS      ns7.ehiweb.it.
                        NS      ns8.ehiweb.it.
0-g                     MX      10 mx01.nic
                        MX      10 mx02.nic
                        MX      10 mx03.nic
                        MX      10 mx04.nic
$ORIGIN 0-g.name
moutamassey             NS      ns01.yahoodomains.jp.
                        NS      ns02.yahoodomains.jp.
`
	to := ParseZone(strings.NewReader(zone), "", "testzone")
	for x := range to {
		fmt.Println(x.RR)
	}
	// Output:
	// name.	3600	IN	SOA	a6.nstld.com. hostmaster.nic.name. 203362132 300 300 1209600 300
	// name.	10800	IN	NS	name.
	// name.	10800	IN	NS	g6.nstld.com.
	// name.	7200	IN	NS	h6.nstld.com.
	// name.	3600	IN	NS	j6.nstld.com.
	// name.	3600	IN	NS	k6.nstld.com.
	// name.	10800	IN	NS	l6.nstld.com.
	// name.	10800	IN	NS	a6.nstld.com.
	// name.	10800	IN	NS	c6.nstld.com.
	// name.	10800	IN	NS	d6.nstld.com.
	// name.	10800	IN	NS	f6.nstld.com.
	// name.	10800	IN	NS	m6.nstld.com.
	// name.	10800	IN	NS	m7.nstld.com.
	// 0-0onlus.name.	10800	IN	NS	ns7.ehiweb.it.
	// 0-0onlus.name.	10800	IN	NS	ns8.ehiweb.it.
	// 0-g.name.	10800	IN	MX	10 mx01.nic.name.
	// 0-g.name.	10800	IN	MX	10 mx02.nic.name.
	// 0-g.name.	10800	IN	MX	10 mx03.nic.name.
	// 0-g.name.	10800	IN	MX	10 mx04.nic.name.
	// moutamassey.0-g.name.name.	10800	IN	NS	ns01.yahoodomains.jp.
	// moutamassey.0-g.name.name.	10800	IN	NS	ns02.yahoodomains.jp.
}

func ExampleHIP() {
	h := `www.example.com     IN  HIP ( 2 200100107B1A74DF365639CC39F1D578
                AwEAAbdxyhNuSutc5EMzxTs9LBPCIkOFH8cIvM4p
9+LrV4e19WzK00+CI6zBCQTdtWsuxKbWIy87UOoJTwkUs7lBu+Upr1gsNrut79ryra+bSRGQ
b1slImA8YVJyuIDsj7kwzG7jnERNqnWxZ48AWkskmdHaVDP4BcelrTI3rMXdXF5D
        rvs.example.com. )`
	if hip, err := NewRR(h); err == nil {
		fmt.Println(hip.String())
	}
	// Output:
	// www.example.com.	3600	IN	HIP	2 200100107B1A74DF365639CC39F1D578 AwEAAbdxyhNuSutc5EMzxTs9LBPCIkOFH8cIvM4p9+LrV4e19WzK00+CI6zBCQTdtWsuxKbWIy87UOoJTwkUs7lBu+Upr1gsNrut79ryra+bSRGQb1slImA8YVJyuIDsj7kwzG7jnERNqnWxZ48AWkskmdHaVDP4BcelrTI3rMXdXF5D rvs.example.com.
}

func TestHIP(t *testing.T) {
	h := `www.example.com.      IN  HIP ( 2 200100107B1A74DF365639CC39F1D578
                                AwEAAbdxyhNuSutc5EMzxTs9LBPCIkOFH8cIvM4p
9+LrV4e19WzK00+CI6zBCQTdtWsuxKbWIy87UOoJTwkUs7lBu+Upr1gsNrut79ryra+bSRGQ
b1slImA8YVJyuIDsj7kwzG7jnERNqnWxZ48AWkskmdHaVDP4BcelrTI3rMXdXF5D
                                rvs1.example.com.
                                rvs2.example.com. )`
	rr, err := NewRR(h)
	if err != nil {
		t.Fatalf("failed to parse RR: %v", err)
	}
	t.Logf("RR: %s", rr)
	msg := new(Msg)
	msg.Answer = []RR{rr, rr}
	bytes, err := msg.Pack()
	if err != nil {
		t.Fatalf("failed to pack msg: %v", err)
	}
	if err := msg.Unpack(bytes); err != nil {
		t.Fatalf("failed to unpack msg: %v", err)
	}
	if len(msg.Answer) != 2 {
		t.Fatalf("2 answers expected: %v", msg)
	}
	for i, rr := range msg.Answer {
		rr := rr.(*HIP)
		t.Logf("RR: %s", rr)
		if l := len(rr.RendezvousServers); l != 2 {
			t.Fatalf("2 servers expected, only %d in record %d:\n%v", l, i, msg)
		}
		for j, s := range []string{"rvs1.example.com.", "rvs2.example.com."} {
			if rr.RendezvousServers[j] != s {
				t.Fatalf("expected server %d of record %d to be %s:\n%v", j, i, s, msg)
			}
		}
	}
}

func ExampleSOA() {
	s := "example.com. 1000 SOA master.example.com. admin.example.com. 1 4294967294 4294967293 4294967295 100"
	if soa, err := NewRR(s); err == nil {
		fmt.Println(soa.String())
	}
	// Output:
	// example.com.	1000	IN	SOA	master.example.com. admin.example.com. 1 4294967294 4294967293 4294967295 100
}

func TestLineNumberError(t *testing.T) {
	s := "example.com. 1000 SOA master.example.com. admin.example.com. monkey 4294967294 4294967293 4294967295 100"
	if _, err := NewRR(s); err != nil {
		if err.Error() != "dns: bad SOA zone parameter: \"monkey\" at line: 1:68" {
			t.Error("not expecting this error: ", err)
		}
	}
}

// Test with no known RR on the line
func TestLineNumberError2(t *testing.T) {
	tests := map[string]string{
		"example.com. 1000 SO master.example.com. admin.example.com. 1 4294967294 4294967293 4294967295 100": "dns: expecting RR type or class, not this...: \"SO\" at line: 1:21",
		"example.com 1000 IN TALINK a.example.com. b..example.com.":                                          "dns: bad TALINK NextName: \"b..example.com.\" at line: 1:57",
		"example.com 1000 IN TALINK ( a.example.com. b..example.com. )":                                      "dns: bad TALINK NextName: \"b..example.com.\" at line: 1:60",
		`example.com 1000 IN TALINK ( a.example.com.
	bb..example.com. )`: "dns: bad TALINK NextName: \"bb..example.com.\" at line: 2:18",
		// This is a bug, it should report an error on line 1, but the new is already processed.
		`example.com 1000 IN TALINK ( a.example.com.  b...example.com.
	)`: "dns: bad TALINK NextName: \"b...example.com.\" at line: 2:1"}

	for in, errStr := range tests {
		_, err := NewRR(in)
		if err == nil {
			t.Error("err is nil")
		} else {
			if err.Error() != errStr {
				t.Errorf("%s: error should be %s is %v", in, errStr, err)
			}
		}
	}
}

// Test if the calculations are correct
func TestRfc1982(t *testing.T) {
	// If the current time and the timestamp are more than 68 years apart
	// it means the date has wrapped. 0 is 1970

	// fall in the current 68 year span
	strtests := []string{"20120525134203", "19700101000000", "20380119031408"}
	for _, v := range strtests {
		if x, _ := StringToTime(v); v != TimeToString(x) {
			t.Errorf("1982 arithmetic string failure %s (%s:%d)", v, TimeToString(x), x)
		}
	}

	inttests := map[uint32]string{0: "19700101000000",
		1 << 31:   "20380119031408",
		1<<32 - 1: "21060207062815",
	}
	for i, v := range inttests {
		if TimeToString(i) != v {
			t.Errorf("1982 arithmetic int failure %d:%s (%s)", i, v, TimeToString(i))
		}
	}

	// Future tests, these dates get parsed to a date within the current 136 year span
	future := map[string]string{"22680119031408": "20631123173144",
		"19010101121212": "20370206184028",
		"19210101121212": "20570206184028",
		"19500101121212": "20860206184028",
		"19700101000000": "19700101000000",
		"19690101000000": "21050207062816",
		"29210101121212": "21040522212236",
	}
	for from, to := range future {
		x, _ := StringToTime(from)
		y := TimeToString(x)
		if y != to {
			t.Errorf("1982 arithmetic future failure %s:%s (%s)", from, to, y)
		}
	}
}

func TestEmpty(t *testing.T) {
	for range ParseZone(strings.NewReader(""), "", "") {
		t.Errorf("should be empty")
	}
}

func TestLowercaseTokens(t *testing.T) {
	var testrecords = []string{
		"example.org. 300 IN a 1.2.3.4",
		"example.org. 300 in A 1.2.3.4",
		"example.org. 300 in a 1.2.3.4",
		"example.org. 300 a 1.2.3.4",
		"example.org. 300 A 1.2.3.4",
		"example.org. IN a 1.2.3.4",
		"example.org. in A 1.2.3.4",
		"example.org. in a 1.2.3.4",
		"example.org. a 1.2.3.4",
		"example.org. A 1.2.3.4",
		"example.org. a 1.2.3.4",
		"$ORIGIN example.org.\n a 1.2.3.4",
		"$Origin example.org.\n a 1.2.3.4",
		"$origin example.org.\n a 1.2.3.4",
		"example.org. Class1 Type1 1.2.3.4",
	}
	for _, testrr := range testrecords {
		_, err := NewRR(testrr)
		if err != nil {
			t.Errorf("failed to parse %#v, got %v", testrr, err)
		}
	}
}

func ExampleParseZone_generate() {
	// From the manual: http://www.bind9.net/manual/bind/9.3.2/Bv9ARM.ch06.html#id2566761
	zone := "$GENERATE 1-2 0 NS SERVER$.EXAMPLE.\n$GENERATE 1-8 $ CNAME $.0"
	to := ParseZone(strings.NewReader(zone), "0.0.192.IN-ADDR.ARPA.", "")
	for x := range to {
		if x.Error == nil {
			fmt.Println(x.RR.String())
		}
	}
	// Output:
	// 0.0.0.192.IN-ADDR.ARPA.	3600	IN	NS	SERVER1.EXAMPLE.
	// 0.0.0.192.IN-ADDR.ARPA.	3600	IN	NS	SERVER2.EXAMPLE.
	// 1.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	1.0.0.0.192.IN-ADDR.ARPA.
	// 2.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	2.0.0.0.192.IN-ADDR.ARPA.
	// 3.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	3.0.0.0.192.IN-ADDR.ARPA.
	// 4.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	4.0.0.0.192.IN-ADDR.ARPA.
	// 5.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	5.0.0.0.192.IN-ADDR.ARPA.
	// 6.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	6.0.0.0.192.IN-ADDR.ARPA.
	// 7.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	7.0.0.0.192.IN-ADDR.ARPA.
	// 8.0.0.192.IN-ADDR.ARPA.	3600	IN	CNAME	8.0.0.0.192.IN-ADDR.ARPA.
}

func TestSRVPacking(t *testing.T) {
	msg := Msg{}

	things := []string{"1.2.3.4:8484",
		"45.45.45.45:8484",
		"84.84.84.84:8484",
	}

	for i, n := range things {
		h, p, err := net.SplitHostPort(n)
		if err != nil {
			continue
		}
		port := 8484
		tmp, err := strconv.Atoi(p)
		if err == nil {
			port = tmp
		}

		rr := &SRV{
			Hdr: RR_Header{Name: "somename.",
				Rrtype: TypeSRV,
				Class:  ClassINET,
				Ttl:    5},
			Priority: uint16(i),
			Weight:   5,
			Port:     uint16(port),
			Target:   h + ".",
		}

		msg.Answer = append(msg.Answer, rr)
	}

	_, err := msg.Pack()
	if err != nil {
		t.Fatalf("couldn't pack %v: %v", msg, err)
	}
}

func TestParseBackslash(t *testing.T) {
	if r, err := NewRR("nul\\000gap.test.globnix.net. 600 IN	A 192.0.2.10"); err != nil {
		t.Errorf("could not create RR with \\000 in it")
	} else {
		t.Logf("parsed %s", r.String())
	}
	if r, err := NewRR(`nul\000gap.test.globnix.net. 600 IN TXT "Hello\123"`); err != nil {
		t.Errorf("could not create RR with \\000 in it")
	} else {
		t.Logf("parsed %s", r.String())
	}
	if r, err := NewRR(`m\ @\ iek.nl. IN 3600 A 127.0.0.1`); err != nil {
		t.Errorf("could not create RR with \\ and \\@ in it")
	} else {
		t.Logf("parsed %s", r.String())
	}
}

func TestILNP(t *testing.T) {
	tests := []string{
		"host1.example.com.\t3600\tIN\tNID\t10 0014:4fff:ff20:ee64",
		"host1.example.com.\t3600\tIN\tNID\t20 0015:5fff:ff21:ee65",
		"host2.example.com.\t3600\tIN\tNID\t10 0016:6fff:ff22:ee66",
		"host1.example.com.\t3600\tIN\tL32\t10 10.1.2.0",
		"host1.example.com.\t3600\tIN\tL32\t20 10.1.4.0",
		"host2.example.com.\t3600\tIN\tL32\t10 10.1.8.0",
		"host1.example.com.\t3600\tIN\tL64\t10 2001:0DB8:1140:1000",
		"host1.example.com.\t3600\tIN\tL64\t20 2001:0DB8:2140:2000",
		"host2.example.com.\t3600\tIN\tL64\t10 2001:0DB8:4140:4000",
		"host1.example.com.\t3600\tIN\tLP\t10 l64-subnet1.example.com.",
		"host1.example.com.\t3600\tIN\tLP\t10 l64-subnet2.example.com.",
		"host1.example.com.\t3600\tIN\tLP\t20 l32-subnet1.example.com.",
	}
	for _, t1 := range tests {
		r, err := NewRR(t1)
		if err != nil {
			t.Fatalf("an error occurred: %v", err)
		} else {
			if t1 != r.String() {
				t.Fatalf("strings should be equal %s %s", t1, r.String())
			}
		}
	}
}

func TestGposEidNimloc(t *testing.T) {
	dt := map[string]string{
		"444433332222111199990123000000ff. NSAP-PTR foo.bar.com.": "444433332222111199990123000000ff.\t3600\tIN\tNSAP-PTR\tfoo.bar.com.",
		"lillee. IN  GPOS -32.6882 116.8652 10.0":                 "lillee.\t3600\tIN\tGPOS\t-32.6882 116.8652 10.0",
		"hinault. IN GPOS -22.6882 116.8652 250.0":                "hinault.\t3600\tIN\tGPOS\t-22.6882 116.8652 250.0",
		"VENERA.   IN NIMLOC  75234159EAC457800920":               "VENERA.\t3600\tIN\tNIMLOC\t75234159EAC457800920",
		"VAXA.     IN EID     3141592653589793":                   "VAXA.\t3600\tIN\tEID\t3141592653589793",
	}
	for i, o := range dt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestPX(t *testing.T) {
	dt := map[string]string{
		"*.net2.it. IN PX 10 net2.it. PRMD-net2.ADMD-p400.C-it.":      "*.net2.it.\t3600\tIN\tPX\t10 net2.it. PRMD-net2.ADMD-p400.C-it.",
		"ab.net2.it. IN PX 10 ab.net2.it. O-ab.PRMD-net2.ADMDb.C-it.": "ab.net2.it.\t3600\tIN\tPX\t10 ab.net2.it. O-ab.PRMD-net2.ADMDb.C-it.",
	}
	for i, o := range dt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestComment(t *testing.T) {
	// Comments we must see
	comments := map[string]bool{"; this is comment 1": true,
		"; this is comment 4": true, "; this is comment 6": true,
		"; this is comment 7": true, "; this is comment 8": true}
	zone := `
foo. IN A 10.0.0.1 ; this is comment 1
foo. IN A (
	10.0.0.2 ; this is comment2
)
; this is comment3
foo. IN A 10.0.0.3
foo. IN A ( 10.0.0.4 ); this is comment 4

foo. IN A 10.0.0.5
; this is comment5

foo. IN A 10.0.0.6

foo. IN DNSKEY 256 3 5 AwEAAb+8l ; this is comment 6
foo. IN NSEC miek.nl. TXT RRSIG NSEC; this is comment 7
foo. IN TXT "THIS IS TEXT MAN"; this is comment 8
`
	for x := range ParseZone(strings.NewReader(zone), ".", "") {
		if x.Error == nil {
			if x.Comment != "" {
				if _, ok := comments[x.Comment]; !ok {
					t.Errorf("wrong comment %s", x.Comment)
				}
			}
		}
	}
}

func TestEUIxx(t *testing.T) {
	tests := map[string]string{
		"host.example. IN EUI48 00-00-5e-90-01-2a":       "host.example.\t3600\tIN\tEUI48\t00-00-5e-90-01-2a",
		"host.example. IN EUI64 00-00-5e-ef-00-00-00-2a": "host.example.\t3600\tIN\tEUI64\t00-00-5e-ef-00-00-00-2a",
	}
	for i, o := range tests {
		r, err := NewRR(i)
		if err != nil {
			t.Errorf("failed to parse %s: %v", i, err)
		}
		if r.String() != o {
			t.Errorf("want %s, got %s", o, r.String())
		}
	}
}

func TestUserRR(t *testing.T) {
	tests := map[string]string{
		"host.example. IN UID 1234":              "host.example.\t3600\tIN\tUID\t1234",
		"host.example. IN GID 1234556":           "host.example.\t3600\tIN\tGID\t1234556",
		"host.example. IN UINFO \"Miek Gieben\"": "host.example.\t3600\tIN\tUINFO\t\"Miek Gieben\"",
	}
	for i, o := range tests {
		r, err := NewRR(i)
		if err != nil {
			t.Errorf("failed to parse %s: %v", i, err)
		}
		if r.String() != o {
			t.Errorf("want %s, got %s", o, r.String())
		}
	}
}

func TestTXT(t *testing.T) {
	// Test single entry TXT record
	rr, err := NewRR(`_raop._tcp.local. 60 IN TXT "single value"`)
	if err != nil {
		t.Error("failed to parse single value TXT record", err)
	} else if rr, ok := rr.(*TXT); !ok {
		t.Error("wrong type, record should be of type TXT")
	} else {
		if len(rr.Txt) != 1 {
			t.Error("bad size of TXT value:", len(rr.Txt))
		} else if rr.Txt[0] != "single value" {
			t.Error("bad single value")
		}
		if rr.String() != `_raop._tcp.local.	60	IN	TXT	"single value"` {
			t.Error("bad representation of TXT record:", rr.String())
		}
		if rr.len() != 28+1+12 {
			t.Error("bad size of serialized record:", rr.len())
		}
	}

	// Test multi entries TXT record
	rr, err = NewRR(`_raop._tcp.local. 60 IN TXT "a=1" "b=2" "c=3" "d=4"`)
	if err != nil {
		t.Error("failed to parse multi-values TXT record", err)
	} else if rr, ok := rr.(*TXT); !ok {
		t.Error("wrong type, record should be of type TXT")
	} else {
		if len(rr.Txt) != 4 {
			t.Error("bad size of TXT multi-value:", len(rr.Txt))
		} else if rr.Txt[0] != "a=1" || rr.Txt[1] != "b=2" || rr.Txt[2] != "c=3" || rr.Txt[3] != "d=4" {
			t.Error("bad values in TXT records")
		}
		if rr.String() != `_raop._tcp.local.	60	IN	TXT	"a=1" "b=2" "c=3" "d=4"` {
			t.Error("bad representation of TXT multi value record:", rr.String())
		}
		if rr.len() != 28+1+3+1+3+1+3+1+3 {
			t.Error("bad size of serialized multi value record:", rr.len())
		}
	}

	// Test empty-string in TXT record
	rr, err = NewRR(`_raop._tcp.local. 60 IN TXT ""`)
	if err != nil {
		t.Error("failed to parse empty-string TXT record", err)
	} else if rr, ok := rr.(*TXT); !ok {
		t.Error("wrong type, record should be of type TXT")
	} else {
		if len(rr.Txt) != 1 {
			t.Error("bad size of TXT empty-string value:", len(rr.Txt))
		} else if rr.Txt[0] != "" {
			t.Error("bad value for empty-string TXT record")
		}
		if rr.String() != `_raop._tcp.local.	60	IN	TXT	""` {
			t.Error("bad representation of empty-string TXT record:", rr.String())
		}
		if rr.len() != 28+1 {
			t.Error("bad size of serialized record:", rr.len())
		}
	}

	// Test TXT record with chunk larger than 255 bytes, they should be split up, by the parser
	s := ""
	for i := 0; i < 255; i++ {
		s += "a"
	}
	s += "b"
	rr, err = NewRR(`test.local. 60 IN TXT "` + s + `"`)
	if err != nil {
		t.Error("failed to parse empty-string TXT record", err)
	}
	if rr.(*TXT).Txt[1] != "b" {
		t.Errorf("Txt should have two chunk, last one my be 'b', but is %s", rr.(*TXT).Txt[1])
	}
	t.Log(rr.String())
}

func TestTypeXXXX(t *testing.T) {
	_, err := NewRR("example.com IN TYPE1234 \\# 4 aabbccdd")
	if err != nil {
		t.Errorf("failed to parse TYPE1234 RR: %v", err)
	}
	_, err = NewRR("example.com IN TYPE655341 \\# 8 aabbccddaabbccdd")
	if err == nil {
		t.Errorf("this should not work, for TYPE655341")
	}
	_, err = NewRR("example.com IN TYPE1 \\# 4 0a000001")
	if err == nil {
		t.Errorf("this should not work")
	}
}

func TestPTR(t *testing.T) {
	_, err := NewRR("144.2.0.192.in-addr.arpa. 900 IN PTR ilouse03146p0\\(.example.com.")
	if err != nil {
		t.Error("failed to parse ", err)
	}
}

func TestDigit(t *testing.T) {
	tests := map[string]byte{
		"miek\\000.nl. 100 IN TXT \"A\"": 0,
		"miek\\001.nl. 100 IN TXT \"A\"": 1,
		"miek\\254.nl. 100 IN TXT \"A\"": 254,
		"miek\\255.nl. 100 IN TXT \"A\"": 255,
		"miek\\256.nl. 100 IN TXT \"A\"": 0,
		"miek\\257.nl. 100 IN TXT \"A\"": 1,
		"miek\\004.nl. 100 IN TXT \"A\"": 4,
	}
	for s, i := range tests {
		r, err := NewRR(s)
		buf := make([]byte, 40)
		if err != nil {
			t.Fatalf("failed to parse %v", err)
		}
		PackRR(r, buf, 0, nil, false)
		t.Log(buf)
		if buf[5] != i {
			t.Fatalf("5 pos must be %d, is %d", i, buf[5])
		}
		r1, _, _ := UnpackRR(buf, 0)
		if r1.Header().Ttl != 100 {
			t.Fatalf("TTL should %d, is %d", 100, r1.Header().Ttl)
		}
	}
}

func TestParseRRSIGTimestamp(t *testing.T) {
	tests := map[string]bool{
		`miek.nl.  IN RRSIG SOA 8 2 43200 20140210031301 20140111031301 12051 miek.nl. MVZUyrYwq0iZhMFDDnVXD2BvuNiUJjSYlJAgzyAE6CF875BMvvZa+Sb0 RlSCL7WODQSQHhCx/fegHhVVF+Iz8N8kOLrmXD1+jO3Bm6Prl5UhcsPx WTBsg/kmxbp8sR1kvH4oZJtVfakG3iDerrxNaf0sQwhZzyfJQAqpC7pcBoc=`: true,
		`miek.nl.  IN RRSIG SOA 8 2 43200 315565800 4102477800 12051 miek.nl. MVZUyrYwq0iZhMFDDnVXD2BvuNiUJjSYlJAgzyAE6CF875BMvvZa+Sb0 RlSCL7WODQSQHhCx/fegHhVVF+Iz8N8kOLrmXD1+jO3Bm6Prl5UhcsPx WTBsg/kmxbp8sR1kvH4oZJtVfakG3iDerrxNaf0sQwhZzyfJQAqpC7pcBoc=`:          true,
	}
	for r := range tests {
		_, err := NewRR(r)
		if err != nil {
			t.Error(err)
		}
	}
}

func TestTxtEqual(t *testing.T) {
	rr1 := new(TXT)
	rr1.Hdr = RR_Header{Name: ".", Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}
	rr1.Txt = []string{"a\"a", "\"", "b"}
	rr2, _ := NewRR(rr1.String())
	if rr1.String() != rr2.String() {
		// This is not an error, but keep this test.
		t.Errorf("these two TXT records should match:\n%s\n%s", rr1.String(), rr2.String())
	}
	t.Logf("%s\n%s", rr1.String(), rr2.String())
}

func TestTxtLong(t *testing.T) {
	rr1 := new(TXT)
	rr1.Hdr = RR_Header{Name: ".", Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}
	// Make a long txt record, this breaks when sending the packet,
	// but not earlier.
	rr1.Txt = []string{"start-"}
	for i := 0; i < 200; i++ {
		rr1.Txt[0] += "start-"
	}
	str := rr1.String()
	if len(str) < len(rr1.Txt[0]) {
		t.Error("string conversion should work")
	}
}

// Basically, don't crash.
func TestMalformedPackets(t *testing.T) {
	var packets = []string{
		"0021641c0000000100000000000078787878787878787878787303636f6d0000100001",
	}

	// com = 63 6f 6d
	for _, packet := range packets {
		data, _ := hex.DecodeString(packet)
		//		for _, v := range data {
		//			t.Log(v)
		//		}
		var msg Msg
		msg.Unpack(data)
		//		println(msg.String())
	}
}

type algorithm struct {
	name uint8
	bits int
}

func TestNewPrivateKey(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	algorithms := []algorithm{
		{ECDSAP256SHA256, 256},
		{ECDSAP384SHA384, 384},
		{RSASHA1, 1024},
		{RSASHA256, 2048},
		{DSA, 1024},
	}

	for _, algo := range algorithms {
		key := new(DNSKEY)
		key.Hdr.Rrtype = TypeDNSKEY
		key.Hdr.Name = "miek.nl."
		key.Hdr.Class = ClassINET
		key.Hdr.Ttl = 14400
		key.Flags = 256
		key.Protocol = 3
		key.Algorithm = algo.name
		privkey, err := key.Generate(algo.bits)
		if err != nil {
			t.Fatal(err)
		}

		newPrivKey, err := key.NewPrivateKey(key.PrivateKeyString(privkey))
		if err != nil {
			t.Error(key.String())
			t.Error(key.PrivateKeyString(privkey))
			t.Fatal(err)
		}

		switch newPrivKey := newPrivKey.(type) {
		case *rsa.PrivateKey:
			newPrivKey.Precompute()
		}

		if !reflect.DeepEqual(privkey, newPrivKey) {
			t.Errorf("[%v] Private keys differ:\n%#v\n%#v", AlgorithmToString[algo.name], privkey, newPrivKey)
		}
	}
}

// special input test
func TestNewRRSpecial(t *testing.T) {
	var (
		rr     RR
		err    error
		expect string
	)

	rr, err = NewRR("; comment")
	expect = ""
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr != nil {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}

	rr, err = NewRR("")
	expect = ""
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr != nil {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}

	rr, err = NewRR("$ORIGIN foo.")
	expect = ""
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr != nil {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}

	rr, err = NewRR(" ")
	expect = ""
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr != nil {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}

	rr, err = NewRR("\n")
	expect = ""
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr != nil {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}

	rr, err = NewRR("foo. A 1.1.1.1\nbar. A 2.2.2.2")
	expect = "foo.\t3600\tIN\tA\t1.1.1.1"
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if rr == nil || rr.String() != expect {
		t.Errorf("unexpected result: [%s] != [%s]", rr, expect)
	}
}

func TestPrintfVerbsRdata(t *testing.T) {
	x, _ := NewRR("www.miek.nl. IN MX 20 mx.miek.nl.")
	if Field(x, 1) != "20" {
		t.Errorf("should be 20")
	}
	if Field(x, 2) != "mx.miek.nl." {
		t.Errorf("should be mx.miek.nl.")
	}

	x, _ = NewRR("www.miek.nl. IN A 127.0.0.1")
	if Field(x, 1) != "127.0.0.1" {
		t.Errorf("should be 127.0.0.1")
	}

	x, _ = NewRR("www.miek.nl. IN AAAA ::1")
	if Field(x, 1) != "::1" {
		t.Errorf("should be ::1")
	}

	x, _ = NewRR("www.miek.nl. IN NSEC a.miek.nl. A NS SOA MX AAAA")
	if Field(x, 1) != "a.miek.nl." {
		t.Errorf("should be a.miek.nl.")
	}
	if Field(x, 2) != "A NS SOA MX AAAA" {
		t.Errorf("should be A NS SOA MX AAAA")
	}

	x, _ = NewRR("www.miek.nl. IN TXT \"first\" \"second\"")
	if Field(x, 1) != "first second" {
		t.Errorf("should be first second")
	}
	if Field(x, 0) != "" {
		t.Errorf("should be empty")
	}
}

func TestParseTokenOverflow(t *testing.T) {
	_, err := NewRR("_443._tcp.example.org. IN TLSA 0 0 0 308205e8308204d0a00302010202100411de8f53b462f6a5a861b712ec6b59300d06092a864886f70d01010b05003070310b300906035504061302555331153013060355040a130c446967694365727420496e6331193017060355040b13107777772e64696769636572742e636f6d312f302d06035504031326446967694365727420534841322048696768204173737572616e636520536572766572204341301e170d3134313130363030303030305a170d3135313131333132303030305a3081a5310b3009060355040613025553311330110603550408130a43616c69666f726e6961311430120603550407130b4c6f7320416e67656c6573313c303a060355040a1333496e7465726e657420436f72706f726174696f6e20666f722041737369676e6564204e616d657320616e64204e756d6265727331133011060355040b130a546563686e6f6c6f6779311830160603550403130f7777772e6578616d706c652e6f726730820122300d06092a864886f70d01010105000382010f003082010a02820101009e663f52a3d18cb67cdfed547408a4e47e4036538988da2798da3b6655f7240d693ed1cb3fe6d6ad3a9e657ff6efa86b83b0cad24e5d31ff2bf70ec3b78b213f1b4bf61bdc669cbbc07d67154128ca92a9b3cbb4213a836fb823ddd4d7cc04918314d25f06086fa9970ba17e357cca9b458c27eb71760ab95e3f9bc898ae89050ae4d09ba2f7e4259d9ff1e072a6971b18355a8b9e53670c3d5dbdbd283f93a764e71b3a4140ca0746090c08510e2e21078d7d07844bf9c03865b531a0bf2ee766bc401f6451c5a1e6f6fb5d5c1d6a97a0abe91ae8b02e89241e07353909ccd5b41c46de207c06801e08f20713603827f2ae3e68cf15ef881d7e0608f70742e30203010001a382024630820242301f0603551d230418301680145168ff90af0207753cccd9656462a212b859723b301d0603551d0e04160414b000a7f422e9b1ce216117c4c46e7164c8e60c553081810603551d11047a3078820f7777772e6578616d706c652e6f7267820b6578616d706c652e636f6d820b6578616d706c652e656475820b6578616d706c652e6e6574820b6578616d706c652e6f7267820f7777772e6578616d706c652e636f6d820f7777772e6578616d706c652e656475820f7777772e6578616d706c652e6e6574300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b0601050507030230750603551d1f046e306c3034a032a030862e687474703a2f2f63726c332e64696769636572742e636f6d2f736861322d68612d7365727665722d67332e63726c3034a032a030862e687474703a2f2f63726c342e64696769636572742e636f6d2f736861322d68612d7365727665722d67332e63726c30420603551d20043b3039303706096086480186fd6c0101302a302806082b06010505070201161c68747470733a2f2f7777772e64696769636572742e636f6d2f43505330818306082b0601050507010104773075302406082b060105050730018618687474703a2f2f6f6373702e64696769636572742e636f6d304d06082b060105050730028641687474703a2f2f636163657274732e64696769636572742e636f6d2f446967694365727453484132486967684173737572616e636553657276657243412e637274300c0603551d130101ff04023000300d06092a864886f70d01010b050003820101005eac2124dedb3978a86ff3608406acb542d3cb54cb83facd63aec88144d6a1bf15dbf1f215c4a73e241e582365cba9ea50dd306541653b3513af1a0756c1b2720e8d112b34fb67181efad9c4609bdc670fb025fa6e6d42188161b026cf3089a08369c2f3609fc84bcc3479140c1922ede430ca8dbac2b2a3cdacb305ba15dc7361c4c3a5e6daa99cb446cb221b28078a7a944efba70d96f31ac143d959bccd2fd50e30c325ea2624fb6b6dbe9344dbcf133bfbd5b4e892d635dbf31596451672c6b65ba5ac9b3cddea92b35dab1065cae3c8cb6bb450a62ea2f72ea7c6bdc7b65fa09b012392543734083c7687d243f8d0375304d99ccd2e148966a8637a6797")
	if err == nil {
		t.Fatalf("token overflow should return an error")
	}
	t.Logf("err: %s\n", err)
}

func TestParseTLSA(t *testing.T) {
	lt := []string{
		"_443._tcp.example.org.\t3600\tIN\tTLSA\t1 1 1 c22be239f483c08957bc106219cc2d3ac1a308dfbbdd0a365f17b9351234cf00",
		"_443._tcp.example.org.\t3600\tIN\tTLSA\t2 1 2 4e85f45179e9cd6e0e68e2eb5be2e85ec9b92d91c609caf3ef0315213e3f92ece92c38397a607214de95c7fadc0ad0f1c604a469a0387959745032c0d51492f3",
		"_443._tcp.example.org.\t3600\tIN\tTLSA\t3 0 2 69ec8d2277360b215d0cd956b0e2747108dff34b27d461a41c800629e38ee6c2d1230cc9e8e36711330adc6766e6ff7c5fbb37f106f248337c1a20ad682888d2",
	}
	for _, o := range lt {
		rr, err := NewRR(o)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", o, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseSSHFP(t *testing.T) {
	lt := []string{
		"test.example.org.\t300\tSSHFP\t1 2 (\n" +
			"\t\t\t\t\tBC6533CDC95A79078A39A56EA7635984ED655318ADA9\n" +
			"\t\t\t\t\tB6159E30723665DA95BB )",
		"test.example.org.\t300\tSSHFP\t1 2 ( BC6533CDC  95A79078A39A56EA7635984ED655318AD  A9B6159E3072366 5DA95BB )",
	}
	result := "test.example.org.\t300\tIN\tSSHFP\t1 2 BC6533CDC95A79078A39A56EA7635984ED655318ADA9B6159E30723665DA95BB"
	for _, o := range lt {
		rr, err := NewRR(o)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != result {
			t.Errorf("`%s' should be equal to\n\n`%s', but is     \n`%s'", o, result, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseHINFO(t *testing.T) {
	dt := map[string]string{
		"example.net. HINFO A B": "example.net.	3600	IN	HINFO	\"A\" \"B\"",
		"example.net. HINFO \"A\" \"B\"": "example.net.	3600	IN	HINFO	\"A\" \"B\"",
		"example.net. HINFO A B C D E F": "example.net.	3600	IN	HINFO	\"A\" \"B C D E F\"",
		"example.net. HINFO AB": "example.net.	3600	IN	HINFO	\"AB\" \"\"",
		// "example.net. HINFO PC-Intel-700mhz \"Redhat Linux 7.1\"": "example.net.	3600	IN	HINFO	\"PC-Intel-700mhz\" \"Redhat Linux 7.1\"",
		// This one is recommended in Pro Bind book http://www.zytrax.com/books/dns/ch8/hinfo.html
		// but effectively, even Bind would replace it to correctly formed text when you AXFR
		// TODO: remove this set of comments or figure support for quoted/unquoted combinations in endingToTxtSlice function
	}
	for i, o := range dt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestParseCAA(t *testing.T) {
	lt := map[string]string{
		"example.net.	CAA	0 issue \"symantec.com\"": "example.net.\t3600\tIN\tCAA\t0 issue \"symantec.com\"",
		"example.net.	CAA	0 issuewild \"symantec.com; stuff\"": "example.net.\t3600\tIN\tCAA\t0 issuewild \"symantec.com; stuff\"",
		"example.net.	CAA	128 tbs \"critical\"": "example.net.\t3600\tIN\tCAA\t128 tbs \"critical\"",
		"example.net.	CAA	2 auth \"0>09\\006\\010+\\006\\001\\004\\001\\214y\\002\\003\\001\\006\\009`\\134H\\001e\\003\\004\\002\\001\\004 y\\209\\012\\221r\\220\\156Q\\218\\150\\150{\\166\\245:\\231\\182%\\157:\\133\\179}\\1923r\\238\\151\\255\\128q\\145\\002\\001\\000\"": "example.net.\t3600\tIN\tCAA\t2 auth \"0>09\\006\\010+\\006\\001\\004\\001\\214y\\002\\003\\001\\006\\009`\\134H\\001e\\003\\004\\002\\001\\004 y\\209\\012\\221r\\220\\156Q\\218\\150\\150{\\166\\245:\\231\\182%\\157:\\133\\179}\\1923r\\238\\151\\255\\128q\\145\\002\\001\\000\"",
		"example.net.   TYPE257	0 issue \"symantec.com\"": "example.net.\t3600\tIN\tCAA\t0 issue \"symantec.com\"",
	}
	for i, o := range lt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}

func TestPackCAA(t *testing.T) {
	m := new(Msg)
	record := new(CAA)
	record.Hdr = RR_Header{Name: "example.com.", Rrtype: TypeCAA, Class: ClassINET, Ttl: 0}
	record.Tag = "issue"
	record.Value = "symantec.com"
	record.Flag = 1

	m.Answer = append(m.Answer, record)
	bytes, err := m.Pack()
	if err != nil {
		t.Fatalf("failed to pack msg: %v", err)
	}
	if err := m.Unpack(bytes); err != nil {
		t.Fatalf("failed to unpack msg: %v", err)
	}
	if len(m.Answer) != 1 {
		t.Fatalf("incorrect number of answers unpacked")
	}
	rr := m.Answer[0].(*CAA)
	if rr.Tag != "issue" {
		t.Fatalf("invalid tag for unpacked answer")
	} else if rr.Value != "symantec.com" {
		t.Fatalf("invalid value for unpacked answer")
	} else if rr.Flag != 1 {
		t.Fatalf("invalid flag for unpacked answer")
	}
}

func TestParseURI(t *testing.T) {
	lt := map[string]string{
		"_http._tcp. IN URI   10 1 \"http://www.example.com/path\"": "_http._tcp.\t3600\tIN\tURI\t10 1 \"http://www.example.com/path\"",
		"_http._tcp. IN URI   10 1 \"\"":                            "_http._tcp.\t3600\tIN\tURI\t10 1 \"\"",
	}
	for i, o := range lt {
		rr, err := NewRR(i)
		if err != nil {
			t.Error("failed to parse RR: ", err)
			continue
		}
		if rr.String() != o {
			t.Errorf("`%s' should be equal to\n`%s', but is     `%s'", i, o, rr.String())
		} else {
			t.Logf("RR is OK: `%s'", rr.String())
		}
	}
}
