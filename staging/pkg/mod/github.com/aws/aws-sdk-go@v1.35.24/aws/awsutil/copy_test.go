package awsutil_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awsutil"
)

func ExampleCopy() {
	type Foo struct {
		A int
		B []*string
	}

	// Create the initial value
	str1 := "hello"
	str2 := "bye bye"
	f1 := &Foo{A: 1, B: []*string{&str1, &str2}}

	// Do the copy
	var f2 Foo
	awsutil.Copy(&f2, f1)

	// Print the result
	fmt.Println(awsutil.Prettify(f2))

	// Output:
	// {
	//   A: 1,
	//   B: ["hello","bye bye"]
	// }
}

func TestCopy1(t *testing.T) {
	type Bar struct {
		a *int
		B *int
		c int
		D int
	}
	type Foo struct {
		A int
		B []*string
		C map[string]*int
		D *time.Time
		E *Bar
	}

	// Create the initial value
	str1 := "hello"
	str2 := "bye bye"
	int1 := 1
	int2 := 2
	intPtr1 := 1
	intPtr2 := 2
	now := time.Now()
	f1 := &Foo{
		A: 1,
		B: []*string{&str1, &str2},
		C: map[string]*int{
			"A": &int1,
			"B": &int2,
		},
		D: &now,
		E: &Bar{
			&intPtr1,
			&intPtr2,
			2,
			3,
		},
	}

	// Do the copy
	var f2 Foo
	awsutil.Copy(&f2, f1)

	// Values are equal
	if v1, v2 := f2.A, f1.A; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.B, f1.B; !reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.C, f1.C; !reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.D, f1.D; !v1.Equal(*v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.E.B, f1.E.B; !reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.E.D, f1.E.D; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}

	// But pointers are not!
	str3 := "nothello"
	int3 := 57
	f2.A = 100
	*f2.B[0] = str3
	*f2.C["B"] = int3
	*f2.D = time.Now()
	f2.E.a = &int3
	*f2.E.B = int3
	f2.E.c = 5
	f2.E.D = 5
	if v1, v2 := f2.A, f1.A; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.B, f1.B; reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.C, f1.C; reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.D, f1.D; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.E.a, f1.E.a; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.E.B, f1.E.B; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.E.c, f1.E.c; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.E.D, f1.E.D; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
}

func TestCopyNestedWithUnexported(t *testing.T) {
	type Bar struct {
		a int
		B int
	}
	type Foo struct {
		A string
		B Bar
	}

	f1 := &Foo{A: "string", B: Bar{a: 1, B: 2}}

	var f2 Foo
	awsutil.Copy(&f2, f1)

	// Values match
	if v1, v2 := f2.A, f1.A; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.B, f1.B; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.B.a, f1.B.a; v1 == v2 {
		t.Errorf("expected values to be not equivalent, but received %v", v1)
	}
	if v1, v2 := f2.B.B, f2.B.B; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
}

func TestCopyIgnoreNilMembers(t *testing.T) {
	type Foo struct {
		A *string
		B []string
		C map[string]string
	}

	f := &Foo{}
	if v1 := f.A; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f.B; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f.C; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}

	var f2 Foo
	awsutil.Copy(&f2, f)
	if v1 := f2.A; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f2.B; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f2.C; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}

	fcopy := awsutil.CopyOf(f)
	f3 := fcopy.(*Foo)
	if v1 := f3.A; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f3.B; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1 := f3.C; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
}

func TestCopyPrimitive(t *testing.T) {
	str := "hello"
	var s string
	awsutil.Copy(&s, &str)
	if v1, v2 := "hello", s; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
}

func TestCopyNil(t *testing.T) {
	var s string
	awsutil.Copy(&s, nil)
	if v1, v2 := "", s; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
}

func TestCopyReader(t *testing.T) {
	var buf io.Reader = bytes.NewReader([]byte("hello world"))
	var r io.Reader
	awsutil.Copy(&r, buf)
	b, err := ioutil.ReadAll(r)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if v1, v2 := []byte("hello world"), b; !bytes.Equal(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}

	// empty bytes because this is not a deep copy
	b, err = ioutil.ReadAll(buf)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if v1, v2 := []byte(""), b; !bytes.Equal(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
}

func TestCopyDifferentStructs(t *testing.T) {
	type SrcFoo struct {
		A                int
		B                []*string
		C                map[string]*int
		SrcUnique        string
		SameNameDiffType int
		unexportedPtr    *int
		ExportedPtr      *int
	}
	type DstFoo struct {
		A                int
		B                []*string
		C                map[string]*int
		DstUnique        int
		SameNameDiffType string
		unexportedPtr    *int
		ExportedPtr      *int
	}

	// Create the initial value
	str1 := "hello"
	str2 := "bye bye"
	int1 := 1
	int2 := 2
	f1 := &SrcFoo{
		A: 1,
		B: []*string{&str1, &str2},
		C: map[string]*int{
			"A": &int1,
			"B": &int2,
		},
		SrcUnique:        "unique",
		SameNameDiffType: 1,
		unexportedPtr:    &int1,
		ExportedPtr:      &int2,
	}

	// Do the copy
	var f2 DstFoo
	awsutil.Copy(&f2, f1)

	// Values are equal
	if v1, v2 := f2.A, f1.A; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.B, f1.B; !reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := f2.C, f1.C; !reflect.DeepEqual(v1, v2) {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := "unique", f1.SrcUnique; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := 1, f1.SameNameDiffType; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := 0, f2.DstUnique; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := "", f2.SameNameDiffType; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := int1, *f1.unexportedPtr; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1 := f2.unexportedPtr; v1 != nil {
		t.Errorf("expected nil, but received %v", v1)
	}
	if v1, v2 := int2, *f1.ExportedPtr; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
	if v1, v2 := int2, *f2.ExportedPtr; v1 != v2 {
		t.Errorf("expected values to be equivalent but received %v and %v", v1, v2)
	}
}

func ExampleCopyOf() {
	type Foo struct {
		A int
		B []*string
	}

	// Create the initial value
	str1 := "hello"
	str2 := "bye bye"
	f1 := &Foo{A: 1, B: []*string{&str1, &str2}}

	// Do the copy
	v := awsutil.CopyOf(f1)
	var f2 *Foo = v.(*Foo)

	// Print the result
	fmt.Println(awsutil.Prettify(f2))

	// Output:
	// {
	//   A: 1,
	//   B: ["hello","bye bye"]
	// }
}
