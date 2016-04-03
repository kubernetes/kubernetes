package metrics

import (
	"bufio"
	"net"
	"testing"
	"time"
)

func acceptConn(addr string) net.Conn {
	ln, _ := net.Listen("tcp", addr)
	conn, _ := ln.Accept()
	return conn
}

func TestStatsite_Flatten(t *testing.T) {
	s := &StatsiteSink{}
	flat := s.flattenKey([]string{"a", "b", "c", "d"})
	if flat != "a.b.c.d" {
		t.Fatalf("Bad flat")
	}
}

func TestStatsite_PushFullQueue(t *testing.T) {
	q := make(chan string, 1)
	q <- "full"

	s := &StatsiteSink{metricQueue: q}
	s.pushMetric("omit")

	out := <-q
	if out != "full" {
		t.Fatalf("bad val %v", out)
	}

	select {
	case v := <-q:
		t.Fatalf("bad val %v", v)
	default:
	}
}

func TestStatsite_Conn(t *testing.T) {
	addr := "localhost:7523"
	done := make(chan bool)
	go func() {
		conn := acceptConn(addr)
		reader := bufio.NewReader(conn)

		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("unexpected err %s", err)
		}
		if line != "gauge.val:1.000000|g\n" {
			t.Fatalf("bad line %s", line)
		}

		line, err = reader.ReadString('\n')
		if err != nil {
			t.Fatalf("unexpected err %s", err)
		}
		if line != "key.other:2.000000|kv\n" {
			t.Fatalf("bad line %s", line)
		}

		line, err = reader.ReadString('\n')
		if err != nil {
			t.Fatalf("unexpected err %s", err)
		}
		if line != "counter.me:3.000000|c\n" {
			t.Fatalf("bad line %s", line)
		}

		line, err = reader.ReadString('\n')
		if err != nil {
			t.Fatalf("unexpected err %s", err)
		}
		if line != "sample.slow_thingy:4.000000|ms\n" {
			t.Fatalf("bad line %s", line)
		}

		conn.Close()
		done <- true
	}()
	s, err := NewStatsiteSink(addr)
	if err != nil {
		t.Fatalf("bad error")
	}

	s.SetGauge([]string{"gauge", "val"}, float32(1))
	s.EmitKey([]string{"key", "other"}, float32(2))
	s.IncrCounter([]string{"counter", "me"}, float32(3))
	s.AddSample([]string{"sample", "slow thingy"}, float32(4))

	select {
	case <-done:
		s.Shutdown()
	case <-time.After(3 * time.Second):
		t.Fatalf("timeout")
	}
}
