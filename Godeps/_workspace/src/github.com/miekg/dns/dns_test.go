package dns

import (
	"encoding/hex"
	"net"
	"testing"
)

func TestPackUnpack(t *testing.T) {
	out := new(Msg)
	out.Answer = make([]RR, 1)
	key := new(DNSKEY)
	key = &DNSKEY{Flags: 257, Protocol: 3, Algorithm: RSASHA1}
	key.Hdr = RR_Header{Name: "miek.nl.", Rrtype: TypeDNSKEY, Class: ClassINET, Ttl: 3600}
	key.PublicKey = "AwEAAaHIwpx3w4VHKi6i1LHnTaWeHCL154Jug0Rtc9ji5qwPXpBo6A5sRv7cSsPQKPIwxLpyCrbJ4mr2L0EPOdvP6z6YfljK2ZmTbogU9aSU2fiq/4wjxbdkLyoDVgtO+JsxNN4bjr4WcWhsmk1Hg93FV9ZpkWb0Tbad8DFqNDzr//kZ"

	out.Answer[0] = key
	msg, err := out.Pack()
	if err != nil {
		t.Error("failed to pack msg with DNSKEY")
	}
	in := new(Msg)
	if in.Unpack(msg) != nil {
		t.Error("failed to unpack msg with DNSKEY")
	}

	sig := new(RRSIG)
	sig = &RRSIG{TypeCovered: TypeDNSKEY, Algorithm: RSASHA1, Labels: 2,
		OrigTtl: 3600, Expiration: 4000, Inception: 4000, KeyTag: 34641, SignerName: "miek.nl.",
		Signature: "AwEAAaHIwpx3w4VHKi6i1LHnTaWeHCL154Jug0Rtc9ji5qwPXpBo6A5sRv7cSsPQKPIwxLpyCrbJ4mr2L0EPOdvP6z6YfljK2ZmTbogU9aSU2fiq/4wjxbdkLyoDVgtO+JsxNN4bjr4WcWhsmk1Hg93FV9ZpkWb0Tbad8DFqNDzr//kZ"}
	sig.Hdr = RR_Header{Name: "miek.nl.", Rrtype: TypeRRSIG, Class: ClassINET, Ttl: 3600}

	out.Answer[0] = sig
	msg, err = out.Pack()
	if err != nil {
		t.Error("failed to pack msg with RRSIG")
	}

	if in.Unpack(msg) != nil {
		t.Error("failed to unpack msg with RRSIG")
	}
}

func TestPackUnpack2(t *testing.T) {
	m := new(Msg)
	m.Extra = make([]RR, 1)
	m.Answer = make([]RR, 1)
	dom := "miek.nl."
	rr := new(A)
	rr.Hdr = RR_Header{Name: dom, Rrtype: TypeA, Class: ClassINET, Ttl: 0}
	rr.A = net.IPv4(127, 0, 0, 1)

	x := new(TXT)
	x.Hdr = RR_Header{Name: dom, Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}
	x.Txt = []string{"heelalaollo"}

	m.Extra[0] = x
	m.Answer[0] = rr
	_, err := m.Pack()
	if err != nil {
		t.Error("Packing failed: ", err)
		return
	}
}

func TestPackUnpack3(t *testing.T) {
	m := new(Msg)
	m.Extra = make([]RR, 2)
	m.Answer = make([]RR, 1)
	dom := "miek.nl."
	rr := new(A)
	rr.Hdr = RR_Header{Name: dom, Rrtype: TypeA, Class: ClassINET, Ttl: 0}
	rr.A = net.IPv4(127, 0, 0, 1)

	x1 := new(TXT)
	x1.Hdr = RR_Header{Name: dom, Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}
	x1.Txt = []string{}

	x2 := new(TXT)
	x2.Hdr = RR_Header{Name: dom, Rrtype: TypeTXT, Class: ClassINET, Ttl: 0}
	x2.Txt = []string{"heelalaollo"}

	m.Extra[0] = x1
	m.Extra[1] = x2
	m.Answer[0] = rr
	b, err := m.Pack()
	if err != nil {
		t.Error("packing failed: ", err)
		return
	}

	var unpackMsg Msg
	err = unpackMsg.Unpack(b)
	if err != nil {
		t.Error("unpacking failed")
		return
	}
}

func TestBailiwick(t *testing.T) {
	yes := map[string]string{
		"miek.nl": "ns.miek.nl",
		".":       "miek.nl",
	}
	for parent, child := range yes {
		if !IsSubDomain(parent, child) {
			t.Errorf("%s should be child of %s", child, parent)
			t.Errorf("comparelabels %d", CompareDomainName(parent, child))
			t.Errorf("lenlabels %d %d", CountLabel(parent), CountLabel(child))
		}
	}
	no := map[string]string{
		"www.miek.nl":  "ns.miek.nl",
		"m\\.iek.nl":   "ns.miek.nl",
		"w\\.iek.nl":   "w.iek.nl",
		"p\\\\.iek.nl": "ns.p.iek.nl", // p\\.iek.nl , literal \ in domain name
		"miek.nl":      ".",
	}
	for parent, child := range no {
		if IsSubDomain(parent, child) {
			t.Errorf("%s should not be child of %s", child, parent)
			t.Errorf("comparelabels %d", CompareDomainName(parent, child))
			t.Errorf("lenlabels %d %d", CountLabel(parent), CountLabel(child))
		}
	}
}

func TestPack(t *testing.T) {
	rr := []string{"US.    86400	IN	NSEC	0-.us. NS SOA RRSIG NSEC DNSKEY TYPE65534"}
	m := new(Msg)
	var err error
	m.Answer = make([]RR, 1)
	for _, r := range rr {
		m.Answer[0], err = NewRR(r)
		if err != nil {
			t.Errorf("failed to create RR: %v", err)
			continue
		}
		if _, err := m.Pack(); err != nil {
			t.Errorf("packing failed: %v", err)
		}
	}
	x := new(Msg)
	ns, _ := NewRR("pool.ntp.org.   390 IN  NS  a.ntpns.org")
	ns.(*NS).Ns = "a.ntpns.org"
	x.Ns = append(m.Ns, ns)
	x.Ns = append(m.Ns, ns)
	x.Ns = append(m.Ns, ns)
	// This crashes due to the fact the a.ntpns.org isn't a FQDN
	// How to recover() from a remove panic()?
	if _, err := x.Pack(); err == nil {
		t.Error("packing should fail")
	}
	x.Answer = make([]RR, 1)
	x.Answer[0], err = NewRR(rr[0])
	if _, err := x.Pack(); err == nil {
		t.Error("packing should fail")
	}
	x.Question = make([]Question, 1)
	x.Question[0] = Question{";sd#eddddséâèµâââ¥âxzztsestxssweewwsssstx@s@Zåµe@cn.pool.ntp.org.", TypeA, ClassINET}
	if _, err := x.Pack(); err == nil {
		t.Error("packing should fail")
	}
}

func TestPackNAPTR(t *testing.T) {
	for _, n := range []string{
		`apple.com. IN NAPTR   100 50 "se" "SIP+D2U" "" _sip._udp.apple.com.`,
		`apple.com. IN NAPTR   90 50 "se" "SIP+D2T" "" _sip._tcp.apple.com.`,
		`apple.com. IN NAPTR   50 50 "se" "SIPS+D2T" "" _sips._tcp.apple.com.`,
	} {
		rr, _ := NewRR(n)
		msg := make([]byte, rr.len())
		if off, err := PackRR(rr, msg, 0, nil, false); err != nil {
			t.Errorf("packing failed: %v", err)
			t.Errorf("length %d, need more than %d", rr.len(), off)
		} else {
			t.Logf("buf size needed: %d", off)
		}
	}
}

func TestCompressLength(t *testing.T) {
	m := new(Msg)
	m.SetQuestion("miek.nl", TypeMX)
	ul := m.Len()
	m.Compress = true
	if ul != m.Len() {
		t.Fatalf("should be equal")
	}
}

// Does the predicted length match final packed length?
func TestMsgCompressLength(t *testing.T) {
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		msg.Compress = true
		return msg
	}

	name1 := "12345678901234567890123456789012345.12345678.123."
	rrA, _ := NewRR(name1 + " 3600 IN A 192.0.2.1")
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	tests := []*Msg{
		makeMsg(name1, []RR{rrA}, nil, nil),
		makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)}

	for _, msg := range tests {
		predicted := msg.Len()
		buf, err := msg.Pack()
		if err != nil {
			t.Error(err)
		}
		if predicted < len(buf) {
			t.Errorf("predicted compressed length is wrong: predicted %s (len=%d) %d, actual %d",
				msg.Question[0].Name, len(msg.Answer), predicted, len(buf))
		}
	}
}

func TestMsgLength(t *testing.T) {
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		return msg
	}

	name1 := "12345678901234567890123456789012345.12345678.123."
	rrA, _ := NewRR(name1 + " 3600 IN A 192.0.2.1")
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	tests := []*Msg{
		makeMsg(name1, []RR{rrA}, nil, nil),
		makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)}

	for _, msg := range tests {
		predicted := msg.Len()
		buf, err := msg.Pack()
		if err != nil {
			t.Error(err)
		}
		if predicted < len(buf) {
			t.Errorf("predicted length is wrong: predicted %s (len=%d), actual %d",
				msg.Question[0].Name, predicted, len(buf))
		}
	}
}

func TestMsgLength2(t *testing.T) {
	// Serialized replies
	var testMessages = []string{
		// google.com. IN A?
		"064e81800001000b0004000506676f6f676c6503636f6d0000010001c00c00010001000000050004adc22986c00c00010001000000050004adc22987c00c00010001000000050004adc22988c00c00010001000000050004adc22989c00c00010001000000050004adc2298ec00c00010001000000050004adc22980c00c00010001000000050004adc22981c00c00010001000000050004adc22982c00c00010001000000050004adc22983c00c00010001000000050004adc22984c00c00010001000000050004adc22985c00c00020001000000050006036e7331c00cc00c00020001000000050006036e7332c00cc00c00020001000000050006036e7333c00cc00c00020001000000050006036e7334c00cc0d800010001000000050004d8ef200ac0ea00010001000000050004d8ef220ac0fc00010001000000050004d8ef240ac10e00010001000000050004d8ef260a0000290500000000050000",
		// amazon.com. IN A? (reply has no EDNS0 record)
		// TODO(miek): this one is off-by-one, need to find out why
		//"6de1818000010004000a000806616d617a6f6e03636f6d0000010001c00c000100010000000500044815c2d4c00c000100010000000500044815d7e8c00c00010001000000050004b02062a6c00c00010001000000050004cdfbf236c00c000200010000000500140570646e733408756c747261646e73036f726700c00c000200010000000500150570646e733508756c747261646e7304696e666f00c00c000200010000000500160570646e733608756c747261646e7302636f02756b00c00c00020001000000050014036e7331037033310664796e656374036e657400c00c00020001000000050006036e7332c0cfc00c00020001000000050006036e7333c0cfc00c00020001000000050006036e7334c0cfc00c000200010000000500110570646e733108756c747261646e73c0dac00c000200010000000500080570646e7332c127c00c000200010000000500080570646e7333c06ec0cb00010001000000050004d04e461fc0eb00010001000000050004cc0dfa1fc0fd00010001000000050004d04e471fc10f00010001000000050004cc0dfb1fc12100010001000000050004cc4a6c01c121001c000100000005001020010502f3ff00000000000000000001c13e00010001000000050004cc4a6d01c13e001c0001000000050010261000a1101400000000000000000001",
		// yahoo.com. IN A?
		"fc2d81800001000300070008057961686f6f03636f6d0000010001c00c00010001000000050004628afd6dc00c00010001000000050004628bb718c00c00010001000000050004cebe242dc00c00020001000000050006036e7336c00cc00c00020001000000050006036e7338c00cc00c00020001000000050006036e7331c00cc00c00020001000000050006036e7332c00cc00c00020001000000050006036e7333c00cc00c00020001000000050006036e7334c00cc00c00020001000000050006036e7335c00cc07b0001000100000005000444b48310c08d00010001000000050004448eff10c09f00010001000000050004cb54dd35c0b100010001000000050004628a0b9dc0c30001000100000005000477a0f77cc05700010001000000050004ca2bdfaac06900010001000000050004caa568160000290500000000050000",
		// microsoft.com. IN A?
		"f4368180000100020005000b096d6963726f736f667403636f6d0000010001c00c0001000100000005000440040b25c00c0001000100000005000441373ac9c00c0002000100000005000e036e7331046d736674036e657400c00c00020001000000050006036e7332c04fc00c00020001000000050006036e7333c04fc00c00020001000000050006036e7334c04fc00c00020001000000050006036e7335c04fc04b000100010000000500044137253ec04b001c00010000000500102a010111200500000000000000010001c0650001000100000005000440043badc065001c00010000000500102a010111200600060000000000010001c07700010001000000050004d5c7b435c077001c00010000000500102a010111202000000000000000010001c08900010001000000050004cf2e4bfec089001c00010000000500102404f800200300000000000000010001c09b000100010000000500044137e28cc09b001c00010000000500102a010111200f000100000000000100010000290500000000050000",
		// google.com. IN MX?
		"724b8180000100050004000b06676f6f676c6503636f6d00000f0001c00c000f000100000005000c000a056173706d78016cc00cc00c000f0001000000050009001404616c7431c02ac00c000f0001000000050009001e04616c7432c02ac00c000f0001000000050009002804616c7433c02ac00c000f0001000000050009003204616c7434c02ac00c00020001000000050006036e7332c00cc00c00020001000000050006036e7333c00cc00c00020001000000050006036e7334c00cc00c00020001000000050006036e7331c00cc02a00010001000000050004adc2421bc02a001c00010000000500102a00145040080c01000000000000001bc04200010001000000050004adc2461bc05700010001000000050004adc2451bc06c000100010000000500044a7d8f1bc081000100010000000500044a7d191bc0ca00010001000000050004d8ef200ac09400010001000000050004d8ef220ac0a600010001000000050004d8ef240ac0b800010001000000050004d8ef260a0000290500000000050000",
		// reddit.com. IN A?
		"12b98180000100080000000c0672656464697403636f6d0000020001c00c0002000100000005000f046175733204616b616d036e657400c00c000200010000000500070475736534c02dc00c000200010000000500070475737733c02dc00c000200010000000500070475737735c02dc00c00020001000000050008056173696131c02dc00c00020001000000050008056173696139c02dc00c00020001000000050008056e73312d31c02dc00c0002000100000005000a076e73312d313935c02dc02800010001000000050004c30a242ec04300010001000000050004451f1d39c05600010001000000050004451f3bc7c0690001000100000005000460073240c07c000100010000000500046007fb81c090000100010000000500047c283484c090001c00010000000500102a0226f0006700000000000000000064c0a400010001000000050004c16c5b01c0a4001c000100000005001026001401000200000000000000000001c0b800010001000000050004c16c5bc3c0b8001c0001000000050010260014010002000000000000000000c30000290500000000050000",
	}

	for i, hexData := range testMessages {
		// we won't fail the decoding of the hex
		input, _ := hex.DecodeString(hexData)
		m := new(Msg)
		m.Unpack(input)
		//println(m.String())
		m.Compress = true
		lenComp := m.Len()
		b, _ := m.Pack()
		pacComp := len(b)
		m.Compress = false
		lenUnComp := m.Len()
		b, _ = m.Pack()
		pacUnComp := len(b)
		if pacComp+1 != lenComp {
			t.Errorf("msg.Len(compressed)=%d actual=%d for test %d", lenComp, pacComp, i)
		}
		if pacUnComp+1 != lenUnComp {
			t.Errorf("msg.Len(uncompressed)=%d actual=%d for test %d", lenUnComp, pacUnComp, i)
		}
	}
}

func TestMsgLengthCompressionMalformed(t *testing.T) {
	// SOA with empty hostmaster, which is illegal
	soa := &SOA{Hdr: RR_Header{Name: ".", Rrtype: TypeSOA, Class: ClassINET, Ttl: 12345},
		Ns:      ".",
		Mbox:    "",
		Serial:  0,
		Refresh: 28800,
		Retry:   7200,
		Expire:  604800,
		Minttl:  60}
	m := new(Msg)
	m.Compress = true
	m.Ns = []RR{soa}
	m.Len() // Should not crash.
}

func BenchmarkMsgLength(b *testing.B) {
	b.StopTimer()
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		msg.Compress = true
		return msg
	}
	name1 := "12345678901234567890123456789012345.12345678.123."
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	msg := makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		msg.Len()
	}
}

func BenchmarkMsgLengthPack(b *testing.B) {
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		msg.Compress = true
		return msg
	}
	name1 := "12345678901234567890123456789012345.12345678.123."
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	msg := makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = msg.Pack()
	}
}

func BenchmarkMsgPackBuffer(b *testing.B) {
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		msg.Compress = true
		return msg
	}
	name1 := "12345678901234567890123456789012345.12345678.123."
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	msg := makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)
	buf := make([]byte, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = msg.PackBuffer(buf)
	}
}

func BenchmarkMsgUnpack(b *testing.B) {
	makeMsg := func(question string, ans, ns, e []RR) *Msg {
		msg := new(Msg)
		msg.SetQuestion(Fqdn(question), TypeANY)
		msg.Answer = append(msg.Answer, ans...)
		msg.Ns = append(msg.Ns, ns...)
		msg.Extra = append(msg.Extra, e...)
		msg.Compress = true
		return msg
	}
	name1 := "12345678901234567890123456789012345.12345678.123."
	rrMx, _ := NewRR(name1 + " 3600 IN MX 10 " + name1)
	msg := makeMsg(name1, []RR{rrMx, rrMx}, nil, nil)
	msgBuf, _ := msg.Pack()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = msg.Unpack(msgBuf)
	}
}

func BenchmarkPackDomainName(b *testing.B) {
	name1 := "12345678901234567890123456789012345.12345678.123."
	buf := make([]byte, len(name1)+1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = PackDomainName(name1, buf, 0, nil, false)
	}
}

func BenchmarkUnpackDomainName(b *testing.B) {
	name1 := "12345678901234567890123456789012345.12345678.123."
	buf := make([]byte, len(name1)+1)
	_, _ = PackDomainName(name1, buf, 0, nil, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = UnpackDomainName(buf, 0)
	}
}

func BenchmarkUnpackDomainNameUnprintable(b *testing.B) {
	name1 := "\x02\x02\x02\x025\x02\x02\x02\x02.12345678.123."
	buf := make([]byte, len(name1)+1)
	_, _ = PackDomainName(name1, buf, 0, nil, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = UnpackDomainName(buf, 0)
	}
}

func TestToRFC3597(t *testing.T) {
	a, _ := NewRR("miek.nl. IN A 10.0.1.1")
	x := new(RFC3597)
	x.ToRFC3597(a)
	if x.String() != `miek.nl.	3600	CLASS1	TYPE1	\# 4 0a000101` {
		t.Error("string mismatch")
	}
}

func TestNoRdataPack(t *testing.T) {
	data := make([]byte, 1024)
	for typ, fn := range typeToRR {
		r := fn()
		*r.Header() = RR_Header{Name: "miek.nl.", Rrtype: typ, Class: ClassINET, Ttl: 3600}
		_, err := PackRR(r, data, 0, nil, false)
		if err != nil {
			t.Errorf("failed to pack RR with zero rdata: %s: %v", TypeToString[typ], err)
		}
	}
}

// TODO(miek): fix dns buffer too small errors this throws
func TestNoRdataUnpack(t *testing.T) {
	data := make([]byte, 1024)
	for typ, fn := range typeToRR {
		if typ == TypeSOA || typ == TypeTSIG || typ == TypeWKS {
			// SOA, TSIG will not be seen (like this) in dyn. updates?
			// WKS is an bug, but...deprecated record.
			continue
		}
		r := fn()
		*r.Header() = RR_Header{Name: "miek.nl.", Rrtype: typ, Class: ClassINET, Ttl: 3600}
		off, err := PackRR(r, data, 0, nil, false)
		if err != nil {
			// Should always works, TestNoDataPack should have caught this
			t.Errorf("failed to pack RR: %v", err)
			continue
		}
		rr, _, err := UnpackRR(data[:off], 0)
		if err != nil {
			t.Errorf("failed to unpack RR with zero rdata: %s: %v", TypeToString[typ], err)
		}
		t.Log(rr)
	}
}

func TestRdataOverflow(t *testing.T) {
	rr := new(RFC3597)
	rr.Hdr.Name = "."
	rr.Hdr.Class = ClassINET
	rr.Hdr.Rrtype = 65280
	rr.Rdata = hex.EncodeToString(make([]byte, 0xFFFF))
	buf := make([]byte, 0xFFFF*2)
	if _, err := PackRR(rr, buf, 0, nil, false); err != nil {
		t.Fatalf("maximum size rrdata pack failed: %v", err)
	}
	rr.Rdata += "00"
	if _, err := PackRR(rr, buf, 0, nil, false); err != ErrRdata {
		t.Fatalf("oversize rrdata pack didn't return ErrRdata - instead: %v", err)
	}
}

func TestCopy(t *testing.T) {
	rr, _ := NewRR("miek.nl. 2311 IN A 127.0.0.1") // Weird TTL to avoid catching TTL
	rr1 := Copy(rr)
	if rr.String() != rr1.String() {
		t.Fatalf("Copy() failed %s != %s", rr.String(), rr1.String())
	}
}

func TestMsgCopy(t *testing.T) {
	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeA)
	rr, _ := NewRR("miek.nl. 2311 IN A 127.0.0.1")
	m.Answer = []RR{rr}
	rr, _ = NewRR("miek.nl. 2311 IN NS 127.0.0.1")
	m.Ns = []RR{rr}

	m1 := m.Copy()
	if m.String() != m1.String() {
		t.Fatalf("Msg.Copy() failed %s != %s", m.String(), m1.String())
	}

	m1.Answer[0], _ = NewRR("somethingelse.nl. 2311 IN A 127.0.0.1")
	if m.String() == m1.String() {
		t.Fatalf("Msg.Copy() failed; change to copy changed template %s", m.String())
	}

	rr, _ = NewRR("miek.nl. 2311 IN A 127.0.0.2")
	m1.Answer = append(m1.Answer, rr)
	if m1.Ns[0].String() == m1.Answer[1].String() {
		t.Fatalf("Msg.Copy() failed; append changed underlying array %s", m1.Ns[0].String())
	}
}

func BenchmarkCopy(b *testing.B) {
	b.ReportAllocs()
	m := new(Msg)
	m.SetQuestion("miek.nl.", TypeA)
	rr, _ := NewRR("miek.nl. 2311 IN A 127.0.0.1")
	m.Answer = []RR{rr}
	rr, _ = NewRR("miek.nl. 2311 IN NS 127.0.0.1")
	m.Ns = []RR{rr}
	rr, _ = NewRR("miek.nl. 2311 IN A 127.0.0.1")
	m.Extra = []RR{rr}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Copy()
	}
}

func TestPackIPSECKEY(t *testing.T) {
	tests := []string{
		"38.2.0.192.in-addr.arpa. 7200 IN     IPSECKEY ( 10 1 2 192.0.2.38 AQNRU3mG7TVTO2BkR47usntb102uFJtugbo6BSGvgqt4AQ== )",
		"38.2.0.192.in-addr.arpa. 7200 IN     IPSECKEY ( 10 0 2 .  AQNRU3mG7TVTO2BkR47usntb102uFJtugbo6BSGvgqt4AQ== )",
		"38.2.0.192.in-addr.arpa. 7200 IN     IPSECKEY ( 10 1 2 192.0.2.3 AQNRU3mG7TVTO2BkR47usntb102uFJtugbo6BSGvgqt4AQ== )",
		"38.1.0.192.in-addr.arpa. 7200 IN     IPSECKEY ( 10 3 2 mygateway.example.com.  AQNRU3mG7TVTO2BkR47usntb102uFJtugbo6BSGvgqt4AQ== )",
		"0.d.4.0.3.0.e.f.f.f.3.f.0.1.2.0 7200 IN     IPSECKEY ( 10 2 2 2001:0DB8:0:8002::2000:1 AQNRU3mG7TVTO2BkR47usntb102uFJtugbo6BSGvgqt4AQ== )",
	}
	buf := make([]byte, 1024)
	for _, t1 := range tests {
		rr, _ := NewRR(t1)
		off, err := PackRR(rr, buf, 0, nil, false)
		if err != nil {
			t.Errorf("failed to pack IPSECKEY %v: %s", err, t1)
			continue
		}

		rr, _, err = UnpackRR(buf[:off], 0)
		if err != nil {
			t.Errorf("failed to unpack IPSECKEY %v: %s", err, t1)
		}
		t.Log(rr)
	}
}

func TestMsgPackBuffer(t *testing.T) {
	var testMessages = []string{
		// news.ycombinator.com.in.escapemg.com.	IN	A, response
		"586285830001000000010000046e6577730b79636f6d62696e61746f7203636f6d02696e086573636170656d6703636f6d0000010001c0210006000100000e10002c036e7332c02103646e730b67726f6f7665736861726bc02d77ed50e600002a3000000e1000093a8000000e10",

		// news.ycombinator.com.in.escapemg.com.	IN	A, question
		"586201000001000000000000046e6577730b79636f6d62696e61746f7203636f6d02696e086573636170656d6703636f6d0000010001",

		"398781020001000000000000046e6577730b79636f6d62696e61746f7203636f6d0000010001",
	}

	for i, hexData := range testMessages {
		// we won't fail the decoding of the hex
		input, _ := hex.DecodeString(hexData)
		m := new(Msg)
		if err := m.Unpack(input); err != nil {
			t.Errorf("packet %d failed to unpack", i)
			continue
		}
		t.Logf("packet %d %s", i, m.String())
	}
}
