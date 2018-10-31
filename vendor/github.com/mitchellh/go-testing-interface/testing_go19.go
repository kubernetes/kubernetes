// +build go1.9

// NOTE: This is a temporary copy of testing.go for Go 1.9 with the addition
// of "Helper" to the T interface. Go 1.9 at the time of typing is in RC
// and is set for release shortly. We'll support this on master as the default
// as soon as 1.9 is released.

package testing

import (
	"fmt"
	"log"
)

// T is the interface that mimics the standard library *testing.T.
//
// In unit tests you can just pass a *testing.T struct. At runtime, outside
// of tests, you can pass in a RuntimeT struct from this package.
type T interface {
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool
	Helper()
}

// RuntimeT implements T and can be instantiated and run at runtime to
// mimic *testing.T behavior. Unlike *testing.T, this will simply panic
// for calls to Fatal. For calls to Error, you'll have to check the errors
// list to determine whether to exit yourself.
type RuntimeT struct {
	skipped bool
	failed  bool
}

func (t *RuntimeT) Error(args ...interface{}) {
	log.Println(fmt.Sprintln(args...))
	t.Fail()
}

func (t *RuntimeT) Errorf(format string, args ...interface{}) {
	log.Printf(format, args...)
	t.Fail()
}

func (t *RuntimeT) Fail() {
	t.failed = true
}

func (t *RuntimeT) FailNow() {
	panic("testing.T failed, see logs for output (if any)")
}

func (t *RuntimeT) Failed() bool {
	return t.failed
}

func (t *RuntimeT) Fatal(args ...interface{}) {
	log.Print(args...)
	t.FailNow()
}

func (t *RuntimeT) Fatalf(format string, args ...interface{}) {
	log.Printf(format, args...)
	t.FailNow()
}

func (t *RuntimeT) Log(args ...interface{}) {
	log.Println(fmt.Sprintln(args...))
}

func (t *RuntimeT) Logf(format string, args ...interface{}) {
	log.Println(fmt.Sprintf(format, args...))
}

func (t *RuntimeT) Name() string {
	return ""
}

func (t *RuntimeT) Skip(args ...interface{}) {
	log.Print(args...)
	t.SkipNow()
}

func (t *RuntimeT) SkipNow() {
	t.skipped = true
}

func (t *RuntimeT) Skipf(format string, args ...interface{}) {
	log.Printf(format, args...)
	t.SkipNow()
}

func (t *RuntimeT) Skipped() bool {
	return t.skipped
}

func (t *RuntimeT) Helper() {}
