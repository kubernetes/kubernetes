package units

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestBytesSize(t *testing.T) {
	assertEquals(t, "1 KiB", BytesSize(1024))
	assertEquals(t, "1 MiB", BytesSize(1024*1024))
	assertEquals(t, "1 MiB", BytesSize(1048576))
	assertEquals(t, "2 MiB", BytesSize(2*MiB))
	assertEquals(t, "3.42 GiB", BytesSize(3.42*GiB))
	assertEquals(t, "5.372 TiB", BytesSize(5.372*TiB))
	assertEquals(t, "2.22 PiB", BytesSize(2.22*PiB))
}

func TestHumanSize(t *testing.T) {
	assertEquals(t, "1 kB", HumanSize(1000))
	assertEquals(t, "1.024 kB", HumanSize(1024))
	assertEquals(t, "1 MB", HumanSize(1000000))
	assertEquals(t, "1.049 MB", HumanSize(1048576))
	assertEquals(t, "2 MB", HumanSize(2*MB))
	assertEquals(t, "3.42 GB", HumanSize(float64(3.42*GB)))
	assertEquals(t, "5.372 TB", HumanSize(float64(5.372*TB)))
	assertEquals(t, "2.22 PB", HumanSize(float64(2.22*PB)))
}

func TestFromHumanSize(t *testing.T) {
	assertSuccessEquals(t, 32, FromHumanSize, "32")
	assertSuccessEquals(t, 32, FromHumanSize, "32b")
	assertSuccessEquals(t, 32, FromHumanSize, "32B")
	assertSuccessEquals(t, 32*KB, FromHumanSize, "32k")
	assertSuccessEquals(t, 32*KB, FromHumanSize, "32K")
	assertSuccessEquals(t, 32*KB, FromHumanSize, "32kb")
	assertSuccessEquals(t, 32*KB, FromHumanSize, "32Kb")
	assertSuccessEquals(t, 32*MB, FromHumanSize, "32Mb")
	assertSuccessEquals(t, 32*GB, FromHumanSize, "32Gb")
	assertSuccessEquals(t, 32*TB, FromHumanSize, "32Tb")
	assertSuccessEquals(t, 32*PB, FromHumanSize, "32Pb")

	assertError(t, FromHumanSize, "")
	assertError(t, FromHumanSize, "hello")
	assertError(t, FromHumanSize, "-32")
	assertError(t, FromHumanSize, "32.3")
	assertError(t, FromHumanSize, " 32 ")
	assertError(t, FromHumanSize, "32.3Kb")
	assertError(t, FromHumanSize, "32 mb")
	assertError(t, FromHumanSize, "32m b")
	assertError(t, FromHumanSize, "32bm")
}

func TestRAMInBytes(t *testing.T) {
	assertSuccessEquals(t, 32, RAMInBytes, "32")
	assertSuccessEquals(t, 32, RAMInBytes, "32b")
	assertSuccessEquals(t, 32, RAMInBytes, "32B")
	assertSuccessEquals(t, 32*KiB, RAMInBytes, "32k")
	assertSuccessEquals(t, 32*KiB, RAMInBytes, "32K")
	assertSuccessEquals(t, 32*KiB, RAMInBytes, "32kb")
	assertSuccessEquals(t, 32*KiB, RAMInBytes, "32Kb")
	assertSuccessEquals(t, 32*MiB, RAMInBytes, "32Mb")
	assertSuccessEquals(t, 32*GiB, RAMInBytes, "32Gb")
	assertSuccessEquals(t, 32*TiB, RAMInBytes, "32Tb")
	assertSuccessEquals(t, 32*PiB, RAMInBytes, "32Pb")
	assertSuccessEquals(t, 32*PiB, RAMInBytes, "32PB")
	assertSuccessEquals(t, 32*PiB, RAMInBytes, "32P")

	assertError(t, RAMInBytes, "")
	assertError(t, RAMInBytes, "hello")
	assertError(t, RAMInBytes, "-32")
	assertError(t, RAMInBytes, "32.3")
	assertError(t, RAMInBytes, " 32 ")
	assertError(t, RAMInBytes, "32.3Kb")
	assertError(t, RAMInBytes, "32 mb")
	assertError(t, RAMInBytes, "32m b")
	assertError(t, RAMInBytes, "32bm")
}

func assertEquals(t *testing.T, expected, actual interface{}) {
	if expected != actual {
		t.Errorf("Expected '%v' but got '%v'", expected, actual)
	}
}

// func that maps to the parse function signatures as testing abstraction
type parseFn func(string) (int64, error)

// Define 'String()' for pretty-print
func (fn parseFn) String() string {
	fnName := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
	return fnName[strings.LastIndex(fnName, ".")+1:]
}

func assertSuccessEquals(t *testing.T, expected int64, fn parseFn, arg string) {
	res, err := fn(arg)
	if err != nil || res != expected {
		t.Errorf("%s(\"%s\") -> expected '%d' but got '%d' with error '%v'", fn, arg, expected, res, err)
	}
}

func assertError(t *testing.T, fn parseFn, arg string) {
	res, err := fn(arg)
	if err == nil && res != -1 {
		t.Errorf("%s(\"%s\") -> expected error but got '%d'", fn, arg, res)
	}
}
