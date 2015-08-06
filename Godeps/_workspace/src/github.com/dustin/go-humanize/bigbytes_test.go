package humanize

import (
	"math/big"
	"testing"
)

func TestBigByteParsing(t *testing.T) {
	tests := []struct {
		in  string
		exp uint64
	}{
		{"42", 42},
		{"42MB", 42000000},
		{"42MiB", 44040192},
		{"42mb", 42000000},
		{"42mib", 44040192},
		{"42MIB", 44040192},
		{"42 MB", 42000000},
		{"42 MiB", 44040192},
		{"42 mb", 42000000},
		{"42 mib", 44040192},
		{"42 MIB", 44040192},
		{"42.5MB", 42500000},
		{"42.5MiB", 44564480},
		{"42.5 MB", 42500000},
		{"42.5 MiB", 44564480},
		// No need to say B
		{"42M", 42000000},
		{"42Mi", 44040192},
		{"42m", 42000000},
		{"42mi", 44040192},
		{"42MI", 44040192},
		{"42 M", 42000000},
		{"42 Mi", 44040192},
		{"42 m", 42000000},
		{"42 mi", 44040192},
		{"42 MI", 44040192},
		{"42.5M", 42500000},
		{"42.5Mi", 44564480},
		{"42.5 M", 42500000},
		{"42.5 Mi", 44564480},
		// Large testing, breaks when too much larger than
		// this.
		{"12.5 EB", uint64(12.5 * float64(EByte))},
		{"12.5 E", uint64(12.5 * float64(EByte))},
		{"12.5 EiB", uint64(12.5 * float64(EiByte))},
	}

	for _, p := range tests {
		got, err := ParseBigBytes(p.in)
		if err != nil {
			t.Errorf("Couldn't parse %v: %v", p.in, err)
		} else {
			if got.Uint64() != p.exp {
				t.Errorf("Expected %v for %v, got %v",
					p.exp, p.in, got)
			}
		}
	}
}

func TestBigByteErrors(t *testing.T) {
	got, err := ParseBigBytes("84 JB")
	if err == nil {
		t.Errorf("Expected error, got %v", got)
	}
	got, err = ParseBigBytes("")
	if err == nil {
		t.Errorf("Expected error parsing nothing")
	}
}

func bbyte(in uint64) string {
	return BigBytes((&big.Int{}).SetUint64(in))
}

func bibyte(in uint64) string {
	return BigIBytes((&big.Int{}).SetUint64(in))
}

func TestBigBytes(t *testing.T) {
	testList{
		{"bytes(0)", bbyte(0), "0B"},
		{"bytes(1)", bbyte(1), "1B"},
		{"bytes(803)", bbyte(803), "803B"},
		{"bytes(999)", bbyte(999), "999B"},

		{"bytes(1024)", bbyte(1024), "1.0kB"},
		{"bytes(1MB - 1)", bbyte(MByte - Byte), "1000kB"},

		{"bytes(1MB)", bbyte(1024 * 1024), "1.0MB"},
		{"bytes(1GB - 1K)", bbyte(GByte - KByte), "1000MB"},

		{"bytes(1GB)", bbyte(GByte), "1.0GB"},
		{"bytes(1TB - 1M)", bbyte(TByte - MByte), "1000GB"},

		{"bytes(1TB)", bbyte(TByte), "1.0TB"},
		{"bytes(1PB - 1T)", bbyte(PByte - TByte), "999TB"},

		{"bytes(1PB)", bbyte(PByte), "1.0PB"},
		{"bytes(1PB - 1T)", bbyte(EByte - PByte), "999PB"},

		{"bytes(1EB)", bbyte(EByte), "1.0EB"},
		// Overflows.
		// {"bytes(1EB - 1P)", Bytes((KByte*EByte)-PByte), "1023EB"},

		{"bytes(0)", bibyte(0), "0B"},
		{"bytes(1)", bibyte(1), "1B"},
		{"bytes(803)", bibyte(803), "803B"},
		{"bytes(1023)", bibyte(1023), "1023B"},

		{"bytes(1024)", bibyte(1024), "1.0KiB"},
		{"bytes(1MB - 1)", bibyte(MiByte - IByte), "1024KiB"},

		{"bytes(1MB)", bibyte(1024 * 1024), "1.0MiB"},
		{"bytes(1GB - 1K)", bibyte(GiByte - KiByte), "1024MiB"},

		{"bytes(1GB)", bibyte(GiByte), "1.0GiB"},
		{"bytes(1TB - 1M)", bibyte(TiByte - MiByte), "1024GiB"},

		{"bytes(1TB)", bibyte(TiByte), "1.0TiB"},
		{"bytes(1PB - 1T)", bibyte(PiByte - TiByte), "1023TiB"},

		{"bytes(1PB)", bibyte(PiByte), "1.0PiB"},
		{"bytes(1PB - 1T)", bibyte(EiByte - PiByte), "1023PiB"},

		{"bytes(1EiB)", bibyte(EiByte), "1.0EiB"},
		// Overflows.
		// {"bytes(1EB - 1P)", bibyte((KIByte*EIByte)-PiByte), "1023EB"},

		{"bytes(5.5GiB)", bibyte(5.5 * GiByte), "5.5GiB"},

		{"bytes(5.5GB)", bbyte(5.5 * GByte), "5.5GB"},
	}.validate(t)
}

func TestVeryBigBytes(t *testing.T) {
	b, _ := (&big.Int{}).SetString("15347691069326346944512", 10)
	s := BigBytes(b)
	if s != "15ZB" {
		t.Errorf("Expected 15ZB, got %v", s)
	}
	s = BigIBytes(b)
	if s != "13ZiB" {
		t.Errorf("Expected 13ZiB, got %v", s)
	}

	b, _ = (&big.Int{}).SetString("15716035654990179271180288", 10)
	s = BigBytes(b)
	if s != "16YB" {
		t.Errorf("Expected 16YB, got %v", s)
	}
	s = BigIBytes(b)
	if s != "13YiB" {
		t.Errorf("Expected 13YiB, got %v", s)
	}
}

func TestVeryVeryBigBytes(t *testing.T) {
	b, _ := (&big.Int{}).SetString("16093220510709943573688614912", 10)
	s := BigBytes(b)
	if s != "16093YB" {
		t.Errorf("Expected 16093YB, got %v", s)
	}
	s = BigIBytes(b)
	if s != "13312YiB" {
		t.Errorf("Expected 13312YiB, got %v", s)
	}
}

func TestParseVeryBig(t *testing.T) {
	tests := []struct {
		in  string
		out string
	}{
		{"16ZB", "16000000000000000000000"},
		{"16ZiB", "18889465931478580854784"},
		{"16.5ZB", "16500000000000000000000"},
		{"16.5ZiB", "19479761741837286506496"},
		{"16Z", "16000000000000000000000"},
		{"16Zi", "18889465931478580854784"},
		{"16.5Z", "16500000000000000000000"},
		{"16.5Zi", "19479761741837286506496"},

		{"16YB", "16000000000000000000000000"},
		{"16YiB", "19342813113834066795298816"},
		{"16.5YB", "16500000000000000000000000"},
		{"16.5YiB", "19947276023641381382651904"},
		{"16Y", "16000000000000000000000000"},
		{"16Yi", "19342813113834066795298816"},
		{"16.5Y", "16500000000000000000000000"},
		{"16.5Yi", "19947276023641381382651904"},
	}

	for _, test := range tests {
		x, err := ParseBigBytes(test.in)
		if err != nil {
			t.Errorf("Error parsing %q: %v", test.in, err)
			continue
		}

		if x.String() != test.out {
			t.Errorf("Expected %q for %q, got %v", test.out, test.in, x)
		}
	}
}

func BenchmarkParseBigBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseBigBytes("16.5Z")
	}
}

func BenchmarkBigBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		bibyte(16.5 * GByte)
	}
}
