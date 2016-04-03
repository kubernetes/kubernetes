package dns_test

import (
	"github.com/miekg/dns"
	"strings"
	"testing"
)

const TypeISBN uint16 = 0x0F01

// A crazy new RR type :)
type ISBN struct {
	x string // rdata with 10 or 13 numbers, dashes or spaces allowed
}

func NewISBN() dns.PrivateRdata { return &ISBN{""} }

func (rd *ISBN) Len() int       { return len([]byte(rd.x)) }
func (rd *ISBN) String() string { return rd.x }

func (rd *ISBN) Parse(txt []string) error {
	rd.x = strings.TrimSpace(strings.Join(txt, " "))
	return nil
}

func (rd *ISBN) Pack(buf []byte) (int, error) {
	b := []byte(rd.x)
	n := copy(buf, b)
	if n != len(b) {
		return n, dns.ErrBuf
	}
	return n, nil
}

func (rd *ISBN) Unpack(buf []byte) (int, error) {
	rd.x = string(buf)
	return len(buf), nil
}

func (rd *ISBN) Copy(dest dns.PrivateRdata) error {
	isbn, ok := dest.(*ISBN)
	if !ok {
		return dns.ErrRdata
	}
	isbn.x = rd.x
	return nil
}

var testrecord = strings.Join([]string{"example.org.", "3600", "IN", "ISBN", "12-3 456789-0-123"}, "\t")

func TestPrivateText(t *testing.T) {
	dns.PrivateHandle("ISBN", TypeISBN, NewISBN)
	defer dns.PrivateHandleRemove(TypeISBN)

	rr, err := dns.NewRR(testrecord)
	if err != nil {
		t.Fatal(err)
	}
	if rr.String() != testrecord {
		t.Errorf("record string representation did not match original %#v != %#v", rr.String(), testrecord)
	} else {
		t.Log(rr.String())
	}
}

func TestPrivateByteSlice(t *testing.T) {
	dns.PrivateHandle("ISBN", TypeISBN, NewISBN)
	defer dns.PrivateHandleRemove(TypeISBN)

	rr, err := dns.NewRR(testrecord)
	if err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, 100)
	off, err := dns.PackRR(rr, buf, 0, nil, false)
	if err != nil {
		t.Errorf("got error packing ISBN: %s", err)
	}

	custrr := rr.(*dns.PrivateRR)
	if ln := custrr.Data.Len() + len(custrr.Header().Name) + 11; ln != off {
		t.Errorf("offset is not matching to length of Private RR: %d!=%d", off, ln)
	}

	rr1, off1, err := dns.UnpackRR(buf[:off], 0)
	if err != nil {
		t.Errorf("got error unpacking ISBN: %s", err)
	}

	if off1 != off {
		t.Errorf("Offset after unpacking differs: %d != %d", off1, off)
	}

	if rr1.String() != testrecord {
		t.Errorf("Record string representation did not match original %#v != %#v", rr1.String(), testrecord)
	} else {
		t.Log(rr1.String())
	}
}

const TypeVERSION uint16 = 0x0F02

type VERSION struct {
	x string
}

func NewVersion() dns.PrivateRdata { return &VERSION{""} }

func (rd *VERSION) String() string { return rd.x }
func (rd *VERSION) Parse(txt []string) error {
	rd.x = strings.TrimSpace(strings.Join(txt, " "))
	return nil
}

func (rd *VERSION) Pack(buf []byte) (int, error) {
	b := []byte(rd.x)
	n := copy(buf, b)
	if n != len(b) {
		return n, dns.ErrBuf
	}
	return n, nil
}

func (rd *VERSION) Unpack(buf []byte) (int, error) {
	rd.x = string(buf)
	return len(buf), nil
}

func (rd *VERSION) Copy(dest dns.PrivateRdata) error {
	isbn, ok := dest.(*VERSION)
	if !ok {
		return dns.ErrRdata
	}
	isbn.x = rd.x
	return nil
}

func (rd *VERSION) Len() int {
	return len([]byte(rd.x))
}

var smallzone = `$ORIGIN example.org.
@ SOA	sns.dns.icann.org. noc.dns.icann.org. (
		2014091518 7200 3600 1209600 3600
)
    A   1.2.3.4
ok ISBN 1231-92110-12
go VERSION (
	1.3.1 ; comment
)
www ISBN 1231-92110-16
*  CNAME @
`

func TestPrivateZoneParser(t *testing.T) {
	dns.PrivateHandle("ISBN", TypeISBN, NewISBN)
	dns.PrivateHandle("VERSION", TypeVERSION, NewVersion)
	defer dns.PrivateHandleRemove(TypeISBN)
	defer dns.PrivateHandleRemove(TypeVERSION)

	r := strings.NewReader(smallzone)
	for x := range dns.ParseZone(r, ".", "") {
		if err := x.Error; err != nil {
			t.Fatal(err)
		}
		t.Log(x.RR)
	}
}
