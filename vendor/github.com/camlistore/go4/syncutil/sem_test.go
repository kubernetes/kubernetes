package syncutil_test

import (
	"testing"

	"go4.org/syncutil"
)

func TestSem(t *testing.T) {
	s := syncutil.NewSem(5)

	if err := s.Acquire(2); err != nil {
		t.Fatal(err)
	}
	if err := s.Acquire(2); err != nil {
		t.Fatal(err)
	}

	go func() {
		s.Release(2)
		s.Release(2)
	}()
	if err := s.Acquire(5); err != nil {
		t.Fatal(err)
	}
}

func TestSemErr(t *testing.T) {
	s := syncutil.NewSem(5)
	if err := s.Acquire(6); err == nil {
		t.Fatal("Didn't get expected error for large acquire.")
	}
}
