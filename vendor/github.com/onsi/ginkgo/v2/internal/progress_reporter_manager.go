package internal

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

type ProgressReporterManager struct {
	lock              *sync.Mutex
	progressReporters map[int]func() string
	prCounter         int
}

func NewProgressReporterManager() *ProgressReporterManager {
	return &ProgressReporterManager{
		progressReporters: map[int]func() string{},
		lock:              &sync.Mutex{},
	}
}

func (prm *ProgressReporterManager) AttachProgressReporter(reporter func() string) func() {
	prm.lock.Lock()
	defer prm.lock.Unlock()
	prm.prCounter += 1
	prCounter := prm.prCounter
	prm.progressReporters[prCounter] = reporter

	return func() {
		prm.lock.Lock()
		defer prm.lock.Unlock()
		delete(prm.progressReporters, prCounter)
	}
}

func (prm *ProgressReporterManager) QueryProgressReporters(ctx context.Context, failer *Failer) []string {
	prm.lock.Lock()
	keys := []int{}
	for key := range prm.progressReporters {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	reporters := []func() string{}
	for _, key := range keys {
		reporters = append(reporters, prm.progressReporters[key])
	}
	prm.lock.Unlock()

	if len(reporters) == 0 {
		return nil
	}
	out := []string{}
	for _, reporter := range reporters {
		reportC := make(chan string, 1)
		go func() {
			defer func() {
				e := recover()
				if e != nil {
					failer.Panic(types.NewCodeLocationWithStackTrace(1), e)
					reportC <- "failed to query attached progress reporter"
				}
			}()
			reportC <- reporter()
		}()
		var report string
		select {
		case report = <-reportC:
		case <-ctx.Done():
			return out
		}
		if strings.TrimSpace(report) != "" {
			out = append(out, report)
		}
	}
	return out
}
