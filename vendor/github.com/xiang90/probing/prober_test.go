package probing

import (
	"net/http/httptest"
	"testing"
	"time"
)

var (
	testID = "testID"
)

func TestProbe(t *testing.T) {
	s := httptest.NewServer(NewHandler())

	p := NewProber(nil)
	p.AddHTTP(testID, time.Millisecond, []string{s.URL})
	defer p.Remove(testID)

	time.Sleep(100 * time.Millisecond)
	status, err := p.Status(testID)
	if err != nil {
		t.Fatalf("err = %v, want %v", err, nil)
	}
	if total := status.Total(); total < 50 || total > 150 {
		t.Fatalf("total = %v, want around %v", total, 100)
	}
	if health := status.Health(); health != true {
		t.Fatalf("health = %v, want %v", health, true)
	}

	// become unhealthy
	s.Close()

	time.Sleep(100 * time.Millisecond)
	if total := status.Total(); total < 150 || total > 250 {
		t.Fatalf("total = %v, want around %v", total, 200)
	}
	if loss := status.Loss(); loss < 50 || loss > 150 {
		t.Fatalf("loss = %v, want around %v", loss, 200)
	}
	if health := status.Health(); health != false {
		t.Fatalf("health = %v, want %v", health, false)
	}
}

func TestProbeReset(t *testing.T) {
	s := httptest.NewServer(NewHandler())
	defer s.Close()

	p := NewProber(nil)
	p.AddHTTP(testID, time.Millisecond, []string{s.URL})
	defer p.Remove(testID)

	time.Sleep(100 * time.Millisecond)
	status, err := p.Status(testID)
	if err != nil {
		t.Fatalf("err = %v, want %v", err, nil)
	}
	if total := status.Total(); total < 50 || total > 150 {
		t.Fatalf("total = %v, want around %v", total, 100)
	}
	if health := status.Health(); health != true {
		t.Fatalf("health = %v, want %v", health, true)
	}

	p.Reset(testID)

	time.Sleep(100 * time.Millisecond)
	if total := status.Total(); total < 50 || total > 150 {
		t.Fatalf("total = %v, want around %v", total, 100)
	}
	if health := status.Health(); health != true {
		t.Fatalf("health = %v, want %v", health, true)
	}
}

func TestProbeRemove(t *testing.T) {
	s := httptest.NewServer(NewHandler())
	defer s.Close()

	p := NewProber(nil)
	p.AddHTTP(testID, time.Millisecond, []string{s.URL})

	p.Remove(testID)
	_, err := p.Status(testID)
	if err != ErrNotFound {
		t.Fatalf("err = %v, want %v", err, ErrNotFound)
	}
}
