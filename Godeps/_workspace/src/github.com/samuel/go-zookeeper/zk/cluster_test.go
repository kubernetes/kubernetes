package zk

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

type logWriter struct {
	t *testing.T
	p string
}

func (lw logWriter) Write(b []byte) (int, error) {
	lw.t.Logf("%s%s", lw.p, string(b))
	return len(b), nil
}

func TestBasicCluster(t *testing.T) {
	ts, err := StartTestCluster(3, nil, logWriter{t: t, p: "[ZKERR] "})
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Stop()
	zk1, err := ts.Connect(0)
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	defer zk1.Close()
	zk2, err := ts.Connect(1)
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	defer zk2.Close()

	time.Sleep(time.Second * 5)

	if _, err := zk1.Create("/gozk-test", []byte("foo-cluster"), 0, WorldACL(PermAll)); err != nil {
		t.Fatalf("Create failed on node 1: %+v", err)
	}
	if by, _, err := zk2.Get("/gozk-test"); err != nil {
		t.Fatalf("Get failed on node 2: %+v", err)
	} else if string(by) != "foo-cluster" {
		t.Fatal("Wrong data for node 2")
	}
}

func TestClientClusterFailover(t *testing.T) {
	ts, err := StartTestCluster(3, nil, logWriter{t: t, p: "[ZKERR] "})
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Stop()
	zk, evCh, err := ts.ConnectAll()
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	defer zk.Close()

	hasSession := make(chan string, 1)
	go func() {
		for ev := range evCh {
			if ev.Type == EventSession && ev.State == StateHasSession {
				select {
				case hasSession <- ev.Server:
				default:
				}
			}
		}
	}()

	waitSession := func() string {
		select {
		case srv := <-hasSession:
			return srv
		case <-time.After(time.Second * 8):
			t.Fatal("Failed to connect and get a session")
		}
		return ""
	}

	srv := waitSession()
	if _, err := zk.Create("/gozk-test", []byte("foo-cluster"), 0, WorldACL(PermAll)); err != nil {
		t.Fatalf("Create failed on node 1: %+v", err)
	}

	stopped := false
	for _, s := range ts.Servers {
		if strings.HasSuffix(srv, fmt.Sprintf(":%d", s.Port)) {
			s.Srv.Stop()
			stopped = true
			break
		}
	}
	if !stopped {
		t.Fatal("Failed to stop server")
	}

	waitSession()
	if by, _, err := zk.Get("/gozk-test"); err != nil {
		t.Fatalf("Get failed on node 2: %+v", err)
	} else if string(by) != "foo-cluster" {
		t.Fatal("Wrong data for node 2")
	}
}

func TestWaitForClose(t *testing.T) {
	ts, err := StartTestCluster(1, nil, logWriter{t: t, p: "[ZKERR] "})
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Stop()
	zk, err := ts.Connect(0)
	if err != nil {
		t.Fatalf("Connect returned error: %+v", err)
	}
	timeout := time.After(30 * time.Second)
CONNECTED:
	for {
		select {
		case ev := <-zk.eventChan:
			if ev.State == StateConnected {
				break CONNECTED
			}
		case <-timeout:
			zk.Close()
			t.Fatal("Timeout")
		}
	}
	zk.Close()
	for {
		select {
		case _, ok := <-zk.eventChan:
			if !ok {
				return
			}
		case <-timeout:
			t.Fatal("Timeout")
		}
	}
}

func TestBadSession(t *testing.T) {
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

	if err := zk.Delete("/gozk-test", -1); err != nil && err != ErrNoNode {
		t.Fatalf("Delete returned error: %+v", err)
	}

	zk.conn.Close()
	time.Sleep(time.Millisecond * 100)

	if err := zk.Delete("/gozk-test", -1); err != nil && err != ErrNoNode {
		t.Fatalf("Delete returned error: %+v", err)
	}
}
