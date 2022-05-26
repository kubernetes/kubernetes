package timeutils

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golangci/golangci-lint/pkg/logutils"
)

const noStagesText = "no stages"

type Stopwatch struct {
	name      string
	startedAt time.Time
	log       logutils.Log

	stages map[string]time.Duration
	mu     sync.Mutex
}

func NewStopwatch(name string, log logutils.Log) *Stopwatch {
	return &Stopwatch{
		name:      name,
		startedAt: time.Now(),
		stages:    map[string]time.Duration{},
		log:       log,
	}
}

type stageDuration struct {
	name string
	d    time.Duration
}

func (s *Stopwatch) stageDurationsSorted() []stageDuration {
	stageDurations := make([]stageDuration, 0, len(s.stages))
	for n, d := range s.stages {
		stageDurations = append(stageDurations, stageDuration{
			name: n,
			d:    d,
		})
	}
	sort.Slice(stageDurations, func(i, j int) bool {
		return stageDurations[i].d > stageDurations[j].d
	})
	return stageDurations
}

func (s *Stopwatch) sprintStages() string {
	if len(s.stages) == 0 {
		return noStagesText
	}

	stageDurations := s.stageDurationsSorted()

	stagesStrings := make([]string, 0, len(stageDurations))
	for _, s := range stageDurations {
		stagesStrings = append(stagesStrings, fmt.Sprintf("%s: %s", s.name, s.d))
	}

	return fmt.Sprintf("stages: %s", strings.Join(stagesStrings, ", "))
}

func (s *Stopwatch) sprintTopStages(n int) string {
	if len(s.stages) == 0 {
		return noStagesText
	}

	stageDurations := s.stageDurationsSorted()

	var stagesStrings []string
	for i := 0; i < len(stageDurations) && i < n; i++ {
		s := stageDurations[i]
		stagesStrings = append(stagesStrings, fmt.Sprintf("%s: %s", s.name, s.d))
	}

	return fmt.Sprintf("top %d stages: %s", n, strings.Join(stagesStrings, ", "))
}

func (s *Stopwatch) Print() {
	p := fmt.Sprintf("%s took %s", s.name, time.Since(s.startedAt))
	if len(s.stages) == 0 {
		s.log.Infof("%s", p)
		return
	}

	s.log.Infof("%s with %s", p, s.sprintStages())
}

func (s *Stopwatch) PrintStages() {
	var stagesDuration time.Duration
	for _, s := range s.stages {
		stagesDuration += s
	}
	s.log.Infof("%s took %s with %s", s.name, stagesDuration, s.sprintStages())
}

func (s *Stopwatch) PrintTopStages(n int) {
	var stagesDuration time.Duration
	for _, s := range s.stages {
		stagesDuration += s
	}
	s.log.Infof("%s took %s with %s", s.name, stagesDuration, s.sprintTopStages(n))
}

func (s *Stopwatch) TrackStage(name string, f func()) {
	startedAt := time.Now()
	f()

	s.mu.Lock()
	s.stages[name] += time.Since(startedAt)
	s.mu.Unlock()
}
