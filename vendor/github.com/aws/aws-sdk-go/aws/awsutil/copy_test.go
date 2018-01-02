package awsutil_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/stretchr/testify/assert"
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
	assert.Equal(t, f2.A, f1.A)
	assert.Equal(t, f2.B, f1.B)
	assert.Equal(t, f2.C, f1.C)
	assert.Equal(t, f2.D, f1.D)
	assert.Equal(t, f2.E.B, f1.E.B)
	assert.Equal(t, f2.E.D, f1.E.D)

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
	assert.NotEqual(t, f2.A, f1.A)
	assert.NotEqual(t, f2.B, f1.B)
	assert.NotEqual(t, f2.C, f1.C)
	assert.NotEqual(t, f2.D, f1.D)
	assert.NotEqual(t, f2.E.a, f1.E.a)
	assert.NotEqual(t, f2.E.B, f1.E.B)
	assert.NotEqual(t, f2.E.c, f1.E.c)
	assert.NotEqual(t, f2.E.D, f1.E.D)
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
	assert.Equal(t, f2.A, f1.A)
	assert.NotEqual(t, f2.B, f1.B)
	assert.NotEqual(t, f2.B.a, f1.B.a)
	assert.Equal(t, f2.B.B, f2.B.B)
}

func TestCopyIgnoreNilMembers(t *testing.T) {
	type Foo struct {
		A *string
		B []string
		C map[string]string
	}

	f := &Foo{}
	assert.Nil(t, f.A)
	assert.Nil(t, f.B)
	assert.Nil(t, f.C)

	var f2 Foo
	awsutil.Copy(&f2, f)
	assert.Nil(t, f2.A)
	assert.Nil(t, f2.B)
	assert.Nil(t, f2.C)

	fcopy := awsutil.CopyOf(f)
	f3 := fcopy.(*Foo)
	assert.Nil(t, f3.A)
	assert.Nil(t, f3.B)
	assert.Nil(t, f3.C)
}

func TestCopyPrimitive(t *testing.T) {
	str := "hello"
	var s string
	awsutil.Copy(&s, &str)
	assert.Equal(t, "hello", s)
}

func TestCopyNil(t *testing.T) {
	var s string
	awsutil.Copy(&s, nil)
	assert.Equal(t, "", s)
}

func TestCopyReader(t *testing.T) {
	var buf io.Reader = bytes.NewReader([]byte("hello world"))
	var r io.Reader
	awsutil.Copy(&r, buf)
	b, err := ioutil.ReadAll(r)
	assert.NoError(t, err)
	assert.Equal(t, []byte("hello world"), b)

	// empty bytes because this is not a deep copy
	b, err = ioutil.ReadAll(buf)
	assert.NoError(t, err)
	assert.Equal(t, []byte(""), b)
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
	assert.Equal(t, f2.A, f1.A)
	assert.Equal(t, f2.B, f1.B)
	assert.Equal(t, f2.C, f1.C)
	assert.Equal(t, "unique", f1.SrcUnique)
	assert.Equal(t, 1, f1.SameNameDiffType)
	assert.Equal(t, 0, f2.DstUnique)
	assert.Equal(t, "", f2.SameNameDiffType)
	assert.Equal(t, int1, *f1.unexportedPtr)
	assert.Nil(t, f2.unexportedPtr)
	assert.Equal(t, int2, *f1.ExportedPtr)
	assert.Equal(t, int2, *f2.ExportedPtr)
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
