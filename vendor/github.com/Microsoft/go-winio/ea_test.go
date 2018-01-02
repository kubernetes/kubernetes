package winio

import (
	"io/ioutil"
	"os"
	"reflect"
	"syscall"
	"testing"
	"unsafe"
)

var (
	testEas = []ExtendedAttribute{
		{Name: "foo", Value: []byte("bar")},
		{Name: "fizz", Value: []byte("buzz")},
	}

	testEasEncoded   = []byte{16, 0, 0, 0, 0, 3, 3, 0, 102, 111, 111, 0, 98, 97, 114, 0, 0, 0, 0, 0, 0, 4, 4, 0, 102, 105, 122, 122, 0, 98, 117, 122, 122, 0, 0, 0}
	testEasNotPadded = testEasEncoded[0 : len(testEasEncoded)-3]
	testEasTruncated = testEasEncoded[0:20]
)

func Test_RoundTripEas(t *testing.T) {
	b, err := EncodeExtendedAttributes(testEas)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testEasEncoded, b) {
		t.Fatalf("encoded mismatch %v %v", testEasEncoded, b)
	}
	eas, err := DecodeExtendedAttributes(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testEas, eas) {
		t.Fatalf("mismatch %+v %+v", testEas, eas)
	}
}

func Test_EasDontNeedPaddingAtEnd(t *testing.T) {
	eas, err := DecodeExtendedAttributes(testEasNotPadded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(testEas, eas) {
		t.Fatalf("mismatch %+v %+v", testEas, eas)
	}
}

func Test_TruncatedEasFailCorrectly(t *testing.T) {
	_, err := DecodeExtendedAttributes(testEasTruncated)
	if err == nil {
		t.Fatal("expected error")
	}
}

func Test_NilEasEncodeAndDecodeAsNil(t *testing.T) {
	b, err := EncodeExtendedAttributes(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 0 {
		t.Fatal("expected empty")
	}
	eas, err := DecodeExtendedAttributes(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(eas) != 0 {
		t.Fatal("expected empty")
	}
}

// Test_SetFileEa makes sure that the test buffer is actually parsable by NtSetEaFile.
func Test_SetFileEa(t *testing.T) {
	f, err := ioutil.TempFile("", "winio")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	ntdll := syscall.MustLoadDLL("ntdll.dll")
	ntSetEaFile := ntdll.MustFindProc("NtSetEaFile")
	var iosb [2]uintptr
	r, _, _ := ntSetEaFile.Call(f.Fd(), uintptr(unsafe.Pointer(&iosb[0])), uintptr(unsafe.Pointer(&testEasEncoded[0])), uintptr(len(testEasEncoded)))
	if r != 0 {
		t.Fatalf("NtSetEaFile failed with %08x", r)
	}
}
