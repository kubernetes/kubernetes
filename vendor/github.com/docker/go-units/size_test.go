package units

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func ExampleBytesSize() {
	fmt.Println(BytesSize(1024))
	fmt.Println(BytesSize(1024 * 1024))
	fmt.Println(BytesSize(1048576))
	fmt.Println(BytesSize(2 * MiB))
	fmt.Println(BytesSize(3.42 * GiB))
	fmt.Println(BytesSize(5.372 * TiB))
	fmt.Println(BytesSize(2.22 * PiB))
}

func ExampleHumanSize() {
	fmt.Println(HumanSize(1000))
	fmt.Println(HumanSize(1024))
	fmt.Println(HumanSize(1000000))
	fmt.Println(HumanSize(1048576))
	fmt.Println(HumanSize(2 * MB))
	fmt.Println(HumanSize(float64(3.42 * GB)))
	fmt.Println(HumanSize(float64(5.372 * TB)))
	fmt.Println(HumanSize(float64(2.22 * PB)))
}

func ExampleFromHumanSize() {
	fmt.Println(FromHumanSize("32"))
	fmt.Println(FromHumanSize("32b"))
	fmt.Println(FromHumanSize("32B"))
	fmt.Println(FromHumanSize("32k"))
	fmt.Println(FromHumanSize("32K"))
	fmt.Println(FromHumanSize("32kb"))
	fmt.Println(FromHumanSize("32Kb"))
	fmt.Println(FromHumanSize("32Mb"))
	fmt.Println(FromHumanSize("32Gb"))
	fmt.Println(FromHumanSize("32Tb"))
	fmt.Println(FromHumanSize("32Pb"))
}

func ExampleRAMInBytes() {
	fmt.Println(RAMInBytes("32"))
	fmt.Println(RAMInBytes("32b"))
	fmt.Println(RAMInBytes("32B"))
	fmt.Println(RAMInBytes("32k"))
	fmt.Println(RAMInBytes("32K"))
	fmt.Println(RAMInBytes("32kb"))
	fmt.Println(RAMInBytes("32Kb"))
	fmt.Println(RAMInBytes("32Mb"))
	fmt.Println(RAMInBytes("32Gb"))
	fmt.Println(RAMInBytes("32Tb"))
	fmt.Println(RAMInBytes("32Pb"))
	fmt.Println(RAMInBytes("32PB"))
	fmt.Println(RAMInBytes("32P"))
}

func TestBytesSize(t *testing.T) {
	assertEquals(t, "1KiB", BytesSize(1024))
	assertEquals(t, "1MiB", BytesSize(1024*1024))
	assertEquals(t, "1MiB", BytesSize(1048576))
	assertEquals(t, "2MiB", BytesSize(2*MiB))
	assertEquals(t, "3.42GiB", BytesSize(3.42*GiB))
	assertEquals(t, "5.372TiB", BytesSize(5.372*TiB))
	assertEquals(t, "2.22PiB", BytesSize(2.22*PiB))
	assertEquals(t, "1.049e+06YiB", BytesSize(KiB*KiB*KiB*KiB*KiB*PiB))
}

func TestHumanSize(t *testing.T) {
	assertEquals(t, "1kB", HumanSize(1000))
	assertEquals(t, "1.024kB", HumanSize(1024))
	assertEquals(t, "1MB", HumanSize(1000000))
	assertEquals(t, "1.049MB", HumanSize(1048576))
	assertEquals(t, "2MB", HumanSize(2*MB))
	assertEquals(t, "3.42GB", HumanSize(float64(3.42*GB)))
	assertEquals(t, "5.372TB", HumanSize(float64(5.372*TB)))
	assertEquals(t, "2.22PB", HumanSize(float64(2.22*PB)))
	assertEquals(t, "1e+04YB", HumanSize(float64(10000000000000*PB)))
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

	assertSuccessEquals(t, 32.5*KB, FromHumanSize, "32.5kB")
	assertSuccessEquals(t, 32.5*KB, FromHumanSize, "32.5 kB")
	assertSuccessEquals(t, 32, FromHumanSize, "32.5 B")

	assertError(t, FromHumanSize, "")
	assertError(t, FromHumanSize, "hello")
	assertError(t, FromHumanSize, "-32")
	assertError(t, FromHumanSize, ".3kB")
	assertError(t, FromHumanSize, " 32 ")
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

	assertSuccessEquals(t, 32, RAMInBytes, "32.3")
	tmp := 32.3 * MiB
	assertSuccessEquals(t, int64(tmp), RAMInBytes, "32.3 mb")

	assertError(t, RAMInBytes, "")
	assertError(t, RAMInBytes, "hello")
	assertError(t, RAMInBytes, "-32")
	assertError(t, RAMInBytes, " 32 ")
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
