package lint

import (
	"time"

	"golang.org/x/tools/go/analysis"
)

const (
	StateInitializing = 0
	StateGraph        = 1
	StateProcessing   = 2
	StateCumulative   = 3
)

type Stats struct {
	State uint32

	InitialPackages          uint32
	TotalPackages            uint32
	ProcessedPackages        uint32
	ProcessedInitialPackages uint32
	Problems                 uint32
	ActiveWorkers            uint32
	TotalWorkers             uint32
	PrintAnalyzerMeasurement func(*analysis.Analyzer, *Package, time.Duration)
}

type AnalysisMeasurementKey struct {
	Analysis string
	Pkg      string
}

func (s *Stats) MeasureAnalyzer(analysis *analysis.Analyzer, pkg *Package, d time.Duration) {
	if s.PrintAnalyzerMeasurement != nil {
		s.PrintAnalyzerMeasurement(analysis, pkg, d)
	}
}
