package netns

import (
	"runtime"
	"sync"
	"testing"
)

func TestGetNewSetDelete(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	origns, err := Get()
	if err != nil {
		t.Fatal(err)
	}
	newns, err := New()
	if err != nil {
		t.Fatal(err)
	}
	if origns.Equal(newns) {
		t.Fatal("New ns failed")
	}
	if err := Set(origns); err != nil {
		t.Fatal(err)
	}
	newns.Close()
	if newns.IsOpen() {
		t.Fatal("newns still open after close", newns)
	}
	ns, err := Get()
	if err != nil {
		t.Fatal(err)
	}
	if !ns.Equal(origns) {
		t.Fatal("Reset ns failed", origns, newns, ns)
	}
}

func TestNone(t *testing.T) {
	ns := None()
	if ns.IsOpen() {
		t.Fatal("None ns is open", ns)
	}
}

func TestThreaded(t *testing.T) {
	ncpu := runtime.GOMAXPROCS(-1)
	if ncpu < 2 {
		t.Skip("-cpu=2 or larger required")
	}

	// Lock this thread simply to ensure other threads get used.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	wg := &sync.WaitGroup{}
	for i := 0; i < ncpu; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			TestGetNewSetDelete(t)
		}()
	}
	wg.Wait()
}
