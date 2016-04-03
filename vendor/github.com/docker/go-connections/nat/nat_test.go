package nat

import (
	"testing"
)

func TestParsePort(t *testing.T) {
	var (
		p   int
		err error
	)

	p, err = ParsePort("1234")

	if err != nil || p != 1234 {
		t.Fatal("Parsing '1234' did not succeed")
	}

	// FIXME currently this is a valid port. I don't think it should be.
	// I'm leaving this test commented out until we make a decision.
	// - erikh

	/*
		p, err = ParsePort("0123")

		if err != nil {
		    t.Fatal("Successfully parsed port '0123' to '123'")
		}
	*/

	p, err = ParsePort("asdf")

	if err == nil || p != 0 {
		t.Fatal("Parsing port 'asdf' succeeded")
	}

	p, err = ParsePort("1asdf")

	if err == nil || p != 0 {
		t.Fatal("Parsing port '1asdf' succeeded")
	}
}

func TestParsePortRangeToInt(t *testing.T) {
	var (
		begin int
		end   int
		err   error
	)

	type TestRange struct {
		Range string
		Begin int
		End   int
	}
	validRanges := []TestRange{
		{"1234", 1234, 1234},
		{"1234-1234", 1234, 1234},
		{"1234-1235", 1234, 1235},
		{"8000-9000", 8000, 9000},
		{"0", 0, 0},
		{"0-0", 0, 0},
	}

	for _, r := range validRanges {
		begin, end, err = ParsePortRangeToInt(r.Range)

		if err != nil || begin != r.Begin {
			t.Fatalf("Parsing port range '%s' did not succeed. Expected begin %d, got %d", r.Range, r.Begin, begin)
		}
		if err != nil || end != r.End {
			t.Fatalf("Parsing port range '%s' did not succeed. Expected end %d, got %d", r.Range, r.End, end)
		}
	}

	invalidRanges := []string{
		"asdf",
		"1asdf",
		"9000-8000",
		"9000-",
		"-8000",
		"-8000-",
	}

	for _, r := range invalidRanges {
		begin, end, err = ParsePortRangeToInt(r)

		if err == nil || begin != 0 || end != 0 {
			t.Fatalf("Parsing port range '%s' succeeded", r)
		}
	}
}

func TestPort(t *testing.T) {
	p, err := NewPort("tcp", "1234")

	if err != nil {
		t.Fatalf("tcp, 1234 had a parsing issue: %v", err)
	}

	if string(p) != "1234/tcp" {
		t.Fatal("tcp, 1234 did not result in the string 1234/tcp")
	}

	if p.Proto() != "tcp" {
		t.Fatal("protocol was not tcp")
	}

	if p.Port() != "1234" {
		t.Fatal("port string value was not 1234")
	}

	if p.Int() != 1234 {
		t.Fatal("port int value was not 1234")
	}

	p, err = NewPort("tcp", "asd1234")
	if err == nil {
		t.Fatal("tcp, asd1234 was supposed to fail")
	}

	p, err = NewPort("tcp", "1234-1230")
	if err == nil {
		t.Fatal("tcp, 1234-1230 was supposed to fail")
	}

	p, err = NewPort("tcp", "1234-1242")
	if err != nil {
		t.Fatalf("tcp, 1234-1242 had a parsing issue: %v", err)
	}

	if string(p) != "1234-1242/tcp" {
		t.Fatal("tcp, 1234-1242 did not result in the string 1234-1242/tcp")
	}
}

func TestSplitProtoPort(t *testing.T) {
	var (
		proto string
		port  string
	)

	proto, port = SplitProtoPort("1234/tcp")

	if proto != "tcp" || port != "1234" {
		t.Fatal("Could not split 1234/tcp properly")
	}

	proto, port = SplitProtoPort("")

	if proto != "" || port != "" {
		t.Fatal("parsing an empty string yielded surprising results", proto, port)
	}

	proto, port = SplitProtoPort("1234")

	if proto != "tcp" || port != "1234" {
		t.Fatal("tcp is not the default protocol for portspec '1234'", proto, port)
	}

	proto, port = SplitProtoPort("1234/")

	if proto != "tcp" || port != "1234" {
		t.Fatal("parsing '1234/' yielded:" + port + "/" + proto)
	}

	proto, port = SplitProtoPort("/tcp")

	if proto != "" || port != "" {
		t.Fatal("parsing '/tcp' yielded:" + port + "/" + proto)
	}
}

func TestParsePortSpecs(t *testing.T) {
	var (
		portMap    map[Port]struct{}
		bindingMap map[Port][]PortBinding
		err        error
	)

	portMap, bindingMap, err = ParsePortSpecs([]string{"1234/tcp", "2345/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1234/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2345/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		if len(bindings) != 1 {
			t.Fatalf("%s should have exactly one binding", portspec)
		}

		if bindings[0].HostIP != "" {
			t.Fatalf("HostIP should not be set for %s", portspec)
		}

		if bindings[0].HostPort != "" {
			t.Fatalf("HostPort should not be set for %s", portspec)
		}
	}

	portMap, bindingMap, err = ParsePortSpecs([]string{"1234:1234/tcp", "2345:2345/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1234/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2345/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		_, port := SplitProtoPort(string(portspec))

		if len(bindings) != 1 {
			t.Fatalf("%s should have exactly one binding", portspec)
		}

		if bindings[0].HostIP != "" {
			t.Fatalf("HostIP should not be set for %s", portspec)
		}

		if bindings[0].HostPort != port {
			t.Fatalf("HostPort should be %s for %s", port, portspec)
		}
	}

	portMap, bindingMap, err = ParsePortSpecs([]string{"0.0.0.0:1234:1234/tcp", "0.0.0.0:2345:2345/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1234/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2345/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		_, port := SplitProtoPort(string(portspec))

		if len(bindings) != 1 {
			t.Fatalf("%s should have exactly one binding", portspec)
		}

		if bindings[0].HostIP != "0.0.0.0" {
			t.Fatalf("HostIP is not 0.0.0.0 for %s", portspec)
		}

		if bindings[0].HostPort != port {
			t.Fatalf("HostPort should be %s for %s", port, portspec)
		}
	}

	_, _, err = ParsePortSpecs([]string{"localhost:1234:1234/tcp"})

	if err == nil {
		t.Fatal("Received no error while trying to parse a hostname instead of ip")
	}
}

func TestParsePortSpecsWithRange(t *testing.T) {
	var (
		portMap    map[Port]struct{}
		bindingMap map[Port][]PortBinding
		err        error
	)

	portMap, bindingMap, err = ParsePortSpecs([]string{"1234-1236/tcp", "2345-2347/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1235/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2346/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		if len(bindings) != 1 {
			t.Fatalf("%s should have exactly one binding", portspec)
		}

		if bindings[0].HostIP != "" {
			t.Fatalf("HostIP should not be set for %s", portspec)
		}

		if bindings[0].HostPort != "" {
			t.Fatalf("HostPort should not be set for %s", portspec)
		}
	}

	portMap, bindingMap, err = ParsePortSpecs([]string{"1234-1236:1234-1236/tcp", "2345-2347:2345-2347/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1235/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2346/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		_, port := SplitProtoPort(string(portspec))
		if len(bindings) != 1 {
			t.Fatalf("%s should have exactly one binding", portspec)
		}

		if bindings[0].HostIP != "" {
			t.Fatalf("HostIP should not be set for %s", portspec)
		}

		if bindings[0].HostPort != port {
			t.Fatalf("HostPort should be %s for %s", port, portspec)
		}
	}

	portMap, bindingMap, err = ParsePortSpecs([]string{"0.0.0.0:1234-1236:1234-1236/tcp", "0.0.0.0:2345-2347:2345-2347/udp"})

	if err != nil {
		t.Fatalf("Error while processing ParsePortSpecs: %s", err)
	}

	if _, ok := portMap[Port("1235/tcp")]; !ok {
		t.Fatal("1234/tcp was not parsed properly")
	}

	if _, ok := portMap[Port("2346/udp")]; !ok {
		t.Fatal("2345/udp was not parsed properly")
	}

	for portspec, bindings := range bindingMap {
		_, port := SplitProtoPort(string(portspec))
		if len(bindings) != 1 || bindings[0].HostIP != "0.0.0.0" || bindings[0].HostPort != port {
			t.Fatalf("Expect single binding to port %s but found %s", port, bindings)
		}
	}

	_, _, err = ParsePortSpecs([]string{"localhost:1234-1236:1234-1236/tcp"})

	if err == nil {
		t.Fatal("Received no error while trying to parse a hostname instead of ip")
	}
}

func TestParseNetworkOptsPrivateOnly(t *testing.T) {
	ports, bindings, err := ParsePortSpecs([]string{"192.168.1.100::80"})
	if err != nil {
		t.Fatal(err)
	}
	if len(ports) != 1 {
		t.Logf("Expected 1 got %d", len(ports))
		t.FailNow()
	}
	if len(bindings) != 1 {
		t.Logf("Expected 1 got %d", len(bindings))
		t.FailNow()
	}
	for k := range ports {
		if k.Proto() != "tcp" {
			t.Logf("Expected tcp got %s", k.Proto())
			t.Fail()
		}
		if k.Port() != "80" {
			t.Logf("Expected 80 got %s", k.Port())
			t.Fail()
		}
		b, exists := bindings[k]
		if !exists {
			t.Log("Binding does not exist")
			t.FailNow()
		}
		if len(b) != 1 {
			t.Logf("Expected 1 got %d", len(b))
			t.FailNow()
		}
		s := b[0]
		if s.HostPort != "" {
			t.Logf("Expected \"\" got %s", s.HostPort)
			t.Fail()
		}
		if s.HostIP != "192.168.1.100" {
			t.Fail()
		}
	}
}

func TestParseNetworkOptsPublic(t *testing.T) {
	ports, bindings, err := ParsePortSpecs([]string{"192.168.1.100:8080:80"})
	if err != nil {
		t.Fatal(err)
	}
	if len(ports) != 1 {
		t.Logf("Expected 1 got %d", len(ports))
		t.FailNow()
	}
	if len(bindings) != 1 {
		t.Logf("Expected 1 got %d", len(bindings))
		t.FailNow()
	}
	for k := range ports {
		if k.Proto() != "tcp" {
			t.Logf("Expected tcp got %s", k.Proto())
			t.Fail()
		}
		if k.Port() != "80" {
			t.Logf("Expected 80 got %s", k.Port())
			t.Fail()
		}
		b, exists := bindings[k]
		if !exists {
			t.Log("Binding does not exist")
			t.FailNow()
		}
		if len(b) != 1 {
			t.Logf("Expected 1 got %d", len(b))
			t.FailNow()
		}
		s := b[0]
		if s.HostPort != "8080" {
			t.Logf("Expected 8080 got %s", s.HostPort)
			t.Fail()
		}
		if s.HostIP != "192.168.1.100" {
			t.Fail()
		}
	}
}

func TestParseNetworkOptsPublicNoPort(t *testing.T) {
	ports, bindings, err := ParsePortSpecs([]string{"192.168.1.100"})

	if err == nil {
		t.Logf("Expected error Invalid containerPort")
		t.Fail()
	}
	if ports != nil {
		t.Logf("Expected nil got %s", ports)
		t.Fail()
	}
	if bindings != nil {
		t.Logf("Expected nil got %s", bindings)
		t.Fail()
	}
}

func TestParseNetworkOptsNegativePorts(t *testing.T) {
	ports, bindings, err := ParsePortSpecs([]string{"192.168.1.100:-1:-1"})

	if err == nil {
		t.Fail()
	}
	if len(ports) != 0 {
		t.Logf("Expected nil got %d", len(ports))
		t.Fail()
	}
	if len(bindings) != 0 {
		t.Logf("Expected 0 got %d", len(bindings))
		t.Fail()
	}
}

func TestParseNetworkOptsUdp(t *testing.T) {
	ports, bindings, err := ParsePortSpecs([]string{"192.168.1.100::6000/udp"})
	if err != nil {
		t.Fatal(err)
	}
	if len(ports) != 1 {
		t.Logf("Expected 1 got %d", len(ports))
		t.FailNow()
	}
	if len(bindings) != 1 {
		t.Logf("Expected 1 got %d", len(bindings))
		t.FailNow()
	}
	for k := range ports {
		if k.Proto() != "udp" {
			t.Logf("Expected udp got %s", k.Proto())
			t.Fail()
		}
		if k.Port() != "6000" {
			t.Logf("Expected 6000 got %s", k.Port())
			t.Fail()
		}
		b, exists := bindings[k]
		if !exists {
			t.Log("Binding does not exist")
			t.FailNow()
		}
		if len(b) != 1 {
			t.Logf("Expected 1 got %d", len(b))
			t.FailNow()
		}
		s := b[0]
		if s.HostPort != "" {
			t.Logf("Expected \"\" got %s", s.HostPort)
			t.Fail()
		}
		if s.HostIP != "192.168.1.100" {
			t.Fail()
		}
	}
}
