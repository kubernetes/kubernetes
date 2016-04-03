package zk

import (
	"testing"
	"time"
)

func TestLock(t *testing.T) {
	ts, err := StartTestCluster(1, nil, logWriter{t: t, p: "[ZKERR] "})
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Stop()
	zk, _, err := ts.ConnectAll()
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	defer zk.Close()

	acls := WorldACL(PermAll)

	l := NewLock(zk, "/test", acls)
	if err := l.Lock(); err != nil {
		t.Fatal(err)
	}
	if err := l.Unlock(); err != nil {
		t.Fatal(err)
	}

	val := make(chan int, 3)

	if err := l.Lock(); err != nil {
		t.Fatal(err)
	}

	l2 := NewLock(zk, "/test", acls)
	go func() {
		if err := l2.Lock(); err != nil {
			t.Fatal(err)
		}
		val <- 2
		if err := l2.Unlock(); err != nil {
			t.Fatal(err)
		}
		val <- 3
	}()
	time.Sleep(time.Millisecond * 100)

	val <- 1
	if err := l.Unlock(); err != nil {
		t.Fatal(err)
	}
	if x := <-val; x != 1 {
		t.Fatalf("Expected 1 instead of %d", x)
	}
	if x := <-val; x != 2 {
		t.Fatalf("Expected 2 instead of %d", x)
	}
	if x := <-val; x != 3 {
		t.Fatalf("Expected 3 instead of %d", x)
	}
}

// This tests creating a lock with a path that's more than 1 node deep (e.g. "/test-multi-level/lock"),
// when a part of that path already exists (i.e. "/test-multi-level" node already exists).
func TestMultiLevelLock(t *testing.T) {
	ts, err := StartTestCluster(1, nil, logWriter{t: t, p: "[ZKERR] "})
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Stop()
	zk, _, err := ts.ConnectAll()
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	defer zk.Close()

	acls := WorldACL(PermAll)
	path := "/test-multi-level"
	if p, err := zk.Create(path, []byte{1, 2, 3, 4}, 0, WorldACL(PermAll)); err != nil {
		t.Fatalf("Create returned error: %+v", err)
	} else if p != path {
		t.Fatalf("Create returned different path '%s' != '%s'", p, path)
	}
	l := NewLock(zk, "/test-multi-level/lock", acls)
	defer zk.Delete("/test-multi-level", -1) // Clean up what we've created for this test
	defer zk.Delete("/test-multi-level/lock", -1)
	if err := l.Lock(); err != nil {
		t.Fatal(err)
	}
	if err := l.Unlock(); err != nil {
		t.Fatal(err)
	}
}
