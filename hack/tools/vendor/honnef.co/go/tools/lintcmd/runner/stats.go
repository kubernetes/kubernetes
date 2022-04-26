package runner

import (
	"sync/atomic"
	"time"

	"honnef.co/go/tools/go/loader"

	"golang.org/x/tools/go/analysis"
)

const (
	StateInitializing = iota
	StateLoadPackageGraph
	StateBuildActionGraph
	StateProcessing
	StateFinalizing
)

type Stats struct {
	state                    uint32
	initialPackages          uint32
	totalPackages            uint32
	processedPackages        uint32
	processedInitialPackages uint32

	// optional function to call every time an analyzer has finished analyzing a package.
	PrintAnalyzerMeasurement func(*analysis.Analyzer, *loader.PackageSpec, time.Duration)
}

func (s *Stats) setState(state uint32)    { atomic.StoreUint32(&s.state, state) }
func (s *Stats) State() int               { return int(atomic.LoadUint32(&s.state)) }
func (s *Stats) setInitialPackages(n int) { atomic.StoreUint32(&s.initialPackages, uint32(n)) }
func (s *Stats) InitialPackages() int     { return int(atomic.LoadUint32(&s.initialPackages)) }
func (s *Stats) setTotalPackages(n int)   { atomic.StoreUint32(&s.totalPackages, uint32(n)) }
func (s *Stats) TotalPackages() int       { return int(atomic.LoadUint32(&s.totalPackages)) }

func (s *Stats) finishPackage()         { atomic.AddUint32(&s.processedPackages, 1) }
func (s *Stats) finishInitialPackage()  { atomic.AddUint32(&s.processedInitialPackages, 1) }
func (s *Stats) ProcessedPackages() int { return int(atomic.LoadUint32(&s.processedPackages)) }
func (s *Stats) ProcessedInitialPackages() int {
	return int(atomic.LoadUint32(&s.processedInitialPackages))
}

func (s *Stats) measureAnalyzer(analysis *analysis.Analyzer, pkg *loader.PackageSpec, d time.Duration) {
	if s.PrintAnalyzerMeasurement != nil {
		s.PrintAnalyzerMeasurement(analysis, pkg, d)
	}
}
