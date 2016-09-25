package watch

import (
	"fmt"
	"math"
	"time"

	"github.com/onsi/ginkgo/ginkgo/testsuite"
)

type Suite struct {
	Suite        testsuite.TestSuite
	RunTime      time.Time
	Dependencies Dependencies

	sharedPackageHashes *PackageHashes
}

func NewSuite(suite testsuite.TestSuite, maxDepth int, sharedPackageHashes *PackageHashes) (*Suite, error) {
	deps, err := NewDependencies(suite.Path, maxDepth)
	if err != nil {
		return nil, err
	}

	sharedPackageHashes.Add(suite.Path)
	for dep := range deps.Dependencies() {
		sharedPackageHashes.Add(dep)
	}

	return &Suite{
		Suite:        suite,
		Dependencies: deps,

		sharedPackageHashes: sharedPackageHashes,
	}, nil
}

func (s *Suite) Delta() float64 {
	delta := s.delta(s.Suite.Path, true, 0) * 1000
	for dep, depth := range s.Dependencies.Dependencies() {
		delta += s.delta(dep, false, depth)
	}
	return delta
}

func (s *Suite) MarkAsRunAndRecomputedDependencies(maxDepth int) error {
	s.RunTime = time.Now()

	deps, err := NewDependencies(s.Suite.Path, maxDepth)
	if err != nil {
		return err
	}

	s.sharedPackageHashes.Add(s.Suite.Path)
	for dep := range deps.Dependencies() {
		s.sharedPackageHashes.Add(dep)
	}

	s.Dependencies = deps

	return nil
}

func (s *Suite) Description() string {
	numDeps := len(s.Dependencies.Dependencies())
	pluralizer := "ies"
	if numDeps == 1 {
		pluralizer = "y"
	}
	return fmt.Sprintf("%s [%d dependenc%s]", s.Suite.Path, numDeps, pluralizer)
}

func (s *Suite) delta(packagePath string, includeTests bool, depth int) float64 {
	return math.Max(float64(s.dt(packagePath, includeTests)), 0) / float64(depth+1)
}

func (s *Suite) dt(packagePath string, includeTests bool) time.Duration {
	packageHash := s.sharedPackageHashes.Get(packagePath)
	var modifiedTime time.Time
	if includeTests {
		modifiedTime = packageHash.TestModifiedTime
	} else {
		modifiedTime = packageHash.CodeModifiedTime
	}

	return modifiedTime.Sub(s.RunTime)
}
