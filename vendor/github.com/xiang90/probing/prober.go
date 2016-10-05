package probing

import (
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"
)

var (
	ErrNotFound = errors.New("probing: id not found")
	ErrExist    = errors.New("probing: id exists")
)

type Prober interface {
	AddHTTP(id string, probingInterval time.Duration, endpoints []string) error
	Remove(id string) error
	RemoveAll()
	Reset(id string) error
	Status(id string) (Status, error)
}

type prober struct {
	mu      sync.Mutex
	targets map[string]*status
	tr      http.RoundTripper
}

func NewProber(tr http.RoundTripper) Prober {
	p := &prober{targets: make(map[string]*status)}
	if tr == nil {
		p.tr = http.DefaultTransport
	} else {
		p.tr = tr
	}
	return p
}

func (p *prober) AddHTTP(id string, probingInterval time.Duration, endpoints []string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.targets[id]; ok {
		return ErrExist
	}

	s := &status{stopC: make(chan struct{})}
	p.targets[id] = s

	ticker := time.NewTicker(probingInterval)

	go func() {
		pinned := 0
		for {
			select {
			case <-ticker.C:
				start := time.Now()
				req, err := http.NewRequest("GET", endpoints[pinned], nil)
				if err != nil {
					panic(err)
				}
				resp, err := p.tr.RoundTrip(req)
				if err != nil {
					s.recordFailure(err)
					pinned = (pinned + 1) % len(endpoints)
					continue
				}

				var hh Health
				d := json.NewDecoder(resp.Body)
				err = d.Decode(&hh)
				resp.Body.Close()
				if err != nil || !hh.OK {
					s.recordFailure(err)
					pinned = (pinned + 1) % len(endpoints)
					continue
				}

				s.record(time.Since(start), hh.Now)
			case <-s.stopC:
				ticker.Stop()
				return
			}
		}
	}()

	return nil
}

func (p *prober) Remove(id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	s, ok := p.targets[id]
	if !ok {
		return ErrNotFound
	}
	close(s.stopC)
	delete(p.targets, id)
	return nil
}

func (p *prober) RemoveAll() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, s := range p.targets {
		close(s.stopC)
	}
	p.targets = make(map[string]*status)
}

func (p *prober) Reset(id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	s, ok := p.targets[id]
	if !ok {
		return ErrNotFound
	}
	s.reset()
	return nil
}

func (p *prober) Status(id string) (Status, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	s, ok := p.targets[id]
	if !ok {
		return nil, ErrNotFound
	}
	return s, nil
}
