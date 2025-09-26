package types

import (
	"fmt"

	"github.com/Masterminds/semver/v3"
)

type SemVerFilter func([]string) bool

func MustParseSemVerFilter(input string) SemVerFilter {
	filter, err := ParseSemVerFilter(input)
	if err != nil {
		panic(err)
	}
	return filter
}

func ParseSemVerFilter(filterVersion string) (SemVerFilter, error) {
	if filterVersion == "" {
		return func(_ []string) bool { return true }, nil
	}

	targetVersion, err := semver.NewVersion(filterVersion)
	if err != nil {
		return nil, fmt.Errorf("invalid filter version: %w", err)
	}

	return func(constraints []string) bool {
		// unconstrained specs always run
		if len(constraints) == 0 {
			return true
		}

		for _, constraintStr := range constraints {
			constraint, err := semver.NewConstraint(constraintStr)
			if err != nil {
				return false
			}

			if !constraint.Check(targetVersion) {
				return false
			}
		}

		return true
	}, nil
}

func ValidateAndCleanupSemVerConstraint(semVerConstraint string, cl CodeLocation) (string, error) {
	if len(semVerConstraint) == 0 {
		return "", GinkgoErrors.InvalidEmptySemVerConstraint(cl)
	}
	_, err := semver.NewConstraint(semVerConstraint)
	if err != nil {
		return "", GinkgoErrors.InvalidSemVerConstraint(semVerConstraint, err.Error(), cl)
	}

	return semVerConstraint, nil
}
