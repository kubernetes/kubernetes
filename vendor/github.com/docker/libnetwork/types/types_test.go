package types

import (
	"flag"
	"net"
	"testing"
)

var runningInContainer = flag.Bool("incontainer", false, "Indicates if the test is running in a container")

func TestTransportPortConv(t *testing.T) {
	sform := "tcp/23"
	tp := &TransportPort{Proto: TCP, Port: uint16(23)}

	if sform != tp.String() {
		t.Fatalf("String() method failed")
	}

	rc := new(TransportPort)
	if err := rc.FromString(sform); err != nil {
		t.Fatal(err)
	}
	if !tp.Equal(rc) {
		t.Fatalf("FromString() method failed")
	}
}

func TestTransportPortBindingConv(t *testing.T) {
	sform := "tcp/172.28.30.23:80/112.0.43.56:8001"
	pb := &PortBinding{
		Proto:    TCP,
		IP:       net.IPv4(172, 28, 30, 23),
		Port:     uint16(80),
		HostIP:   net.IPv4(112, 0, 43, 56),
		HostPort: uint16(8001),
	}

	rc := new(PortBinding)
	if err := rc.FromString(sform); err != nil {
		t.Fatal(err)
	}
	if !pb.Equal(rc) {
		t.Fatalf("FromString() method failed")
	}
}

func TestErrorConstructors(t *testing.T) {
	var err error

	err = BadRequestErrorf("Io ho %d uccello", 1)
	if err.Error() != "Io ho 1 uccello" {
		t.Fatal(err)
	}
	if _, ok := err.(BadRequestError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = RetryErrorf("Incy wincy %s went up the spout again", "spider")
	if err.Error() != "Incy wincy spider went up the spout again" {
		t.Fatal(err)
	}
	if _, ok := err.(RetryError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = NotFoundErrorf("Can't find the %s", "keys")
	if err.Error() != "Can't find the keys" {
		t.Fatal(err)
	}
	if _, ok := err.(NotFoundError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = ForbiddenErrorf("Can't open door %d", 2)
	if err.Error() != "Can't open door 2" {
		t.Fatal(err)
	}
	if _, ok := err.(ForbiddenError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = NotImplementedErrorf("Functionality %s is not implemented", "x")
	if err.Error() != "Functionality x is not implemented" {
		t.Fatal(err)
	}
	if _, ok := err.(NotImplementedError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = TimeoutErrorf("Process %s timed out", "abc")
	if err.Error() != "Process abc timed out" {
		t.Fatal(err)
	}
	if _, ok := err.(TimeoutError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = NoServiceErrorf("Driver %s is not available", "mh")
	if err.Error() != "Driver mh is not available" {
		t.Fatal(err)
	}
	if _, ok := err.(NoServiceError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = InternalErrorf("Not sure what happened")
	if err.Error() != "Not sure what happened" {
		t.Fatal(err)
	}
	if _, ok := err.(InternalError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); ok {
		t.Fatal(err)
	}

	err = InternalMaskableErrorf("Minor issue, it can be ignored")
	if err.Error() != "Minor issue, it can be ignored" {
		t.Fatal(err)
	}
	if _, ok := err.(InternalError); !ok {
		t.Fatal(err)
	}
	if _, ok := err.(MaskableError); !ok {
		t.Fatal(err)
	}
}

func TestCompareIPMask(t *testing.T) {
	input := []struct {
		ip    net.IP
		mask  net.IPMask
		is    int
		ms    int
		isErr bool
	}{
		{ // ip in v4Inv6 representation, mask in v4 representation
			ip:   net.IPv4(172, 28, 30, 1),
			mask: []byte{0xff, 0xff, 0xff, 0},
			is:   12,
			ms:   0,
		},
		{ // ip and mask in v4Inv6 representation
			ip:   net.IPv4(172, 28, 30, 2),
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			is:   12,
			ms:   12,
		},
		{ // ip in v4 representation, mask in v4Inv6 representation
			ip:   net.IPv4(172, 28, 30, 3)[12:],
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			is:   0,
			ms:   12,
		},
		{ // ip and mask in v4 representation
			ip:   net.IPv4(172, 28, 30, 4)[12:],
			mask: []byte{0xff, 0xff, 0xff, 0},
			is:   0,
			ms:   0,
		},
		{ // ip and mask as v6
			ip:   net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"),
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0},
			is:   0,
			ms:   0,
		},
		{
			ip:    net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"),
			mask:  []byte{0xff, 0xff, 0xff, 0},
			isErr: true,
		},
		{
			ip:    net.ParseIP("173.32.4.5"),
			mask:  []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0},
			isErr: true,
		},
		{
			ip:    net.ParseIP("173.32.4.5"),
			mask:  []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			isErr: true,
		},
	}

	for ind, i := range input {
		is, ms, err := compareIPMask(i.ip, i.mask)
		if i.isErr {
			if err == nil {
				t.Fatalf("Incorrect error condition for element %d. is: %d, ms: %d, err: %v", ind, is, ms, err)
			}
		} else {
			if i.is != is || i.ms != ms {
				t.Fatalf("expected is: %d, ms: %d. Got is: %d, ms: %d for element %d", i.is, i.ms, is, ms, ind)
			}
		}
	}
}

func TestUtilGetHostPortionIP(t *testing.T) {
	input := []struct {
		ip   net.IP
		mask net.IPMask
		host net.IP
		err  error
	}{
		{ // ip in v4Inv6 representation, mask in v4 representation
			ip:   net.IPv4(172, 28, 30, 1),
			mask: []byte{0xff, 0xff, 0xff, 0},
			host: net.IPv4(0, 0, 0, 1),
		},
		{ // ip and mask in v4Inv6 representation
			ip:   net.IPv4(172, 28, 30, 2),
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			host: net.IPv4(0, 0, 0, 2),
		},
		{ // ip in v4 representation, mask in v4Inv6 representation
			ip:   net.IPv4(172, 28, 30, 3)[12:],
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			host: net.IPv4(0, 0, 0, 3)[12:],
		},
		{ // ip and mask in v4 representation
			ip:   net.IPv4(172, 28, 30, 4)[12:],
			mask: []byte{0xff, 0xff, 0xff, 0},
			host: net.IPv4(0, 0, 0, 4)[12:],
		},
		{ // ip and mask as v6
			ip:   net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"),
			mask: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0},
			host: net.ParseIP("0::AB:00CD"),
		},
	}

	for _, i := range input {
		h, err := GetHostPartIP(i.ip, i.mask)
		if err != nil {
			t.Fatal(err)
		}
		if !i.host.Equal(h) {
			t.Fatalf("Failed to return expected host ip. Expected: %s. Got: %s", i.host, h)
		}
	}

	// ip as v6 and mask as v4 are not compatible
	if _, err := GetHostPartIP(net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"), []byte{0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
	// ip as v4 and non conventional mask
	if _, err := GetHostPartIP(net.ParseIP("173.32.4.5"), []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
	// ip as v4 and non conventional mask
	if _, err := GetHostPartIP(net.ParseIP("173.32.4.5"), []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
}

func TestUtilGetBroadcastIP(t *testing.T) {
	input := []struct {
		ip    net.IP
		mask  net.IPMask
		bcast net.IP
		err   error
	}{
		// ip in v4Inv6 representation, mask in v4 representation
		{
			ip:    net.IPv4(172, 28, 30, 1),
			mask:  []byte{0xff, 0xff, 0xff, 0},
			bcast: net.IPv4(172, 28, 30, 255),
		},
		{
			ip:    net.IPv4(10, 28, 30, 1),
			mask:  []byte{0xff, 0, 0, 0},
			bcast: net.IPv4(10, 255, 255, 255),
		},
		// ip and mask in v4Inv6 representation
		{
			ip:    net.IPv4(172, 28, 30, 2),
			mask:  []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			bcast: net.IPv4(172, 28, 30, 255),
		},
		{
			ip:    net.IPv4(172, 28, 30, 2),
			mask:  []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0},
			bcast: net.IPv4(172, 28, 255, 255),
		},
		// ip in v4 representation, mask in v4Inv6 representation
		{
			ip:    net.IPv4(172, 28, 30, 3)[12:],
			mask:  []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0},
			bcast: net.IPv4(172, 28, 30, 255)[12:],
		},
		{
			ip:    net.IPv4(172, 28, 30, 3)[12:],
			mask:  []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0},
			bcast: net.IPv4(172, 255, 255, 255)[12:],
		},
		// ip and mask in v4 representation
		{
			ip:    net.IPv4(172, 28, 30, 4)[12:],
			mask:  []byte{0xff, 0xff, 0xff, 0},
			bcast: net.IPv4(172, 28, 30, 255)[12:],
		},
		{
			ip:    net.IPv4(172, 28, 30, 4)[12:],
			mask:  []byte{0xff, 0xff, 0, 0},
			bcast: net.IPv4(172, 28, 255, 255)[12:],
		},
		{ // ip and mask as v6
			ip:    net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"),
			mask:  []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0},
			bcast: net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEFF:FFFF"),
		},
	}

	for _, i := range input {
		h, err := GetBroadcastIP(i.ip, i.mask)
		if err != nil {
			t.Fatal(err)
		}
		if !i.bcast.Equal(h) {
			t.Fatalf("Failed to return expected host ip. Expected: %s. Got: %s", i.bcast, h)
		}
	}

	// ip as v6 and mask as v4 are not compatible
	if _, err := GetBroadcastIP(net.ParseIP("2001:DB8:2002:2001:FFFF:ABCD:EEAB:00CD"), []byte{0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
	// ip as v4 and non conventional mask
	if _, err := GetBroadcastIP(net.ParseIP("173.32.4.5"), []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
	// ip as v4 and non conventional mask
	if _, err := GetBroadcastIP(net.ParseIP("173.32.4.5"), []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0}); err == nil {
		t.Fatalf("Unexpected success")
	}
}

func TestParseCIDR(t *testing.T) {
	input := []struct {
		cidr string
		ipnw *net.IPNet
	}{
		{"192.168.22.44/16", &net.IPNet{IP: net.IP{192, 168, 22, 44}, Mask: net.IPMask{255, 255, 0, 0}}},
		{"10.10.2.0/24", &net.IPNet{IP: net.IP{10, 10, 2, 0}, Mask: net.IPMask{255, 255, 255, 0}}},
		{"10.0.0.100/17", &net.IPNet{IP: net.IP{10, 0, 0, 100}, Mask: net.IPMask{255, 255, 128, 0}}},
	}

	for _, i := range input {
		nw, err := ParseCIDR(i.cidr)
		if err != nil {
			t.Fatal(err)
		}
		if !CompareIPNet(nw, i.ipnw) {
			t.Fatalf("network differ. Expected %v. Got: %v", i.ipnw, nw)
		}
	}
}
