package pubsub

import (
	"fmt"
	"testing"
	"time"
)

func TestSendToOneSub(t *testing.T) {
	p := NewPublisher(100*time.Millisecond, 10)
	c := p.Subscribe()

	p.Publish("hi")

	msg := <-c
	if msg.(string) != "hi" {
		t.Fatalf("expected message hi but received %v", msg)
	}
}

func TestSendToMultipleSubs(t *testing.T) {
	p := NewPublisher(100*time.Millisecond, 10)
	subs := []chan interface{}{}
	subs = append(subs, p.Subscribe(), p.Subscribe(), p.Subscribe())

	p.Publish("hi")

	for _, c := range subs {
		msg := <-c
		if msg.(string) != "hi" {
			t.Fatalf("expected message hi but received %v", msg)
		}
	}
}

func TestEvictOneSub(t *testing.T) {
	p := NewPublisher(100*time.Millisecond, 10)
	s1 := p.Subscribe()
	s2 := p.Subscribe()

	p.Evict(s1)
	p.Publish("hi")
	if _, ok := <-s1; ok {
		t.Fatal("expected s1 to not receive the published message")
	}

	msg := <-s2
	if msg.(string) != "hi" {
		t.Fatalf("expected message hi but received %v", msg)
	}
}

func TestClosePublisher(t *testing.T) {
	p := NewPublisher(100*time.Millisecond, 10)
	subs := []chan interface{}{}
	subs = append(subs, p.Subscribe(), p.Subscribe(), p.Subscribe())
	p.Close()

	for _, c := range subs {
		if _, ok := <-c; ok {
			t.Fatal("expected all subscriber channels to be closed")
		}
	}
}

const sampleText = "test"

type testSubscriber struct {
	dataCh chan interface{}
	ch     chan error
}

func (s *testSubscriber) Wait() error {
	return <-s.ch
}

func newTestSubscriber(p *Publisher) *testSubscriber {
	ts := &testSubscriber{
		dataCh: p.Subscribe(),
		ch:     make(chan error),
	}
	go func() {
		for data := range ts.dataCh {
			s, ok := data.(string)
			if !ok {
				ts.ch <- fmt.Errorf("Unexpected type %T", data)
				break
			}
			if s != sampleText {
				ts.ch <- fmt.Errorf("Unexpected text %s", s)
				break
			}
		}
		close(ts.ch)
	}()
	return ts
}

// for testing with -race
func TestPubSubRace(t *testing.T) {
	p := NewPublisher(0, 1024)
	var subs [](*testSubscriber)
	for j := 0; j < 50; j++ {
		subs = append(subs, newTestSubscriber(p))
	}
	for j := 0; j < 1000; j++ {
		p.Publish(sampleText)
	}
	time.AfterFunc(1*time.Second, func() {
		for _, s := range subs {
			p.Evict(s.dataCh)
		}
	})
	for _, s := range subs {
		s.Wait()
	}
}

func BenchmarkPubSub(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		p := NewPublisher(0, 1024)
		var subs [](*testSubscriber)
		for j := 0; j < 50; j++ {
			subs = append(subs, newTestSubscriber(p))
		}
		b.StartTimer()
		for j := 0; j < 1000; j++ {
			p.Publish(sampleText)
		}
		time.AfterFunc(1*time.Second, func() {
			for _, s := range subs {
				p.Evict(s.dataCh)
			}
		})
		for _, s := range subs {
			if err := s.Wait(); err != nil {
				b.Fatal(err)
			}
		}
	}
}
