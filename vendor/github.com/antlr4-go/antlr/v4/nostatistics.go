//go:build !antlr.stats

package antlr

// This file is compiled when the build configuration antlr.stats is not enabled.
// which then allows the compiler to optimize out all the code that is not used.
const collectStats = false

// goRunStats is a dummy struct used when build configuration antlr.stats is not enabled.
type goRunStats struct {
}

var Statistics = &goRunStats{}

func (s *goRunStats) AddJStatRec(_ *JStatRec) {
	// Do nothing - compiler will optimize this out (hopefully)
}

func (s *goRunStats) CollectionAnomalies() {
	// Do nothing - compiler will optimize this out (hopefully)
}

func (s *goRunStats) Reset() {
	// Do nothing - compiler will optimize this out (hopefully)
}

func (s *goRunStats) Report(dir string, prefix string) error {
	// Do nothing - compiler will optimize this out (hopefully)
	return nil
}

func (s *goRunStats) Analyze() {
	// Do nothing - compiler will optimize this out (hopefully)
}

type statsOption func(*goRunStats) error

func (s *goRunStats) Configure(options ...statsOption) error {
	// Do nothing - compiler will optimize this out (hopefully)
	return nil
}

func WithTopN(topN int) statsOption {
	return func(s *goRunStats) error {
		return nil
	}
}
