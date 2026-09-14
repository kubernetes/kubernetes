package types

import (
	"fmt"
	"strings"

	"github.com/Masterminds/semver/v3"
)

type SemVerFilter func(component string, constraints []string) bool

func MustParseSemVerFilter(input string) SemVerFilter {
	filter, err := ParseSemVerFilter(input)
	if err != nil {
		panic(err)
	}
	return filter
}

// ParseSemVerFilter parses non-component and component-specific semantic version filter string.
// The filter string can contain multiple non-component and component-specific versions separated by commas.
// Each component-specific version is in the format "component=version".
// If a version is specified without a component, it applies to non-component-specific constraints.
func ParseSemVerFilter(componentFilterVersions string) (SemVerFilter, error) {
	if componentFilterVersions == "" {
		return func(_ string, _ []string) bool { return true }, nil
	}

	result := map[string]*semver.Version{}
	parts := strings.Split(componentFilterVersions, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if len(part) == 0 {
			continue
		}
		if strings.Contains(part, "=") {
			// validate component-specific version string
			invalidPart, invalidErr := false, fmt.Errorf("invalid component filter version: %s", part)
			subParts := strings.Split(part, "=")
			if len(subParts) != 2 {
				invalidPart = true
			}
			component := strings.TrimSpace(subParts[0])
			versionStr := strings.TrimSpace(subParts[1])
			if len(component) == 0 || len(versionStr) == 0 {
				invalidPart = true
			}
			if invalidPart {
				return nil, invalidErr
			}

			// validate semver
			v, err := semver.NewVersion(versionStr)
			if err != nil {
				return nil, fmt.Errorf("invalid component filter version: %s, error: %w", part, err)
			}
			result[component] = v
		} else {
			v, err := semver.NewVersion(part)
			if err != nil {
				return nil, fmt.Errorf("invalid filter version: %s, error: %w", part, err)
			}
			result[""] = v
		}
	}

	return func(component string, constraints []string) bool {
		// unconstrained specs always run
		if len(component) == 0 && len(constraints) == 0 {
			return true
		}

		// check non-component specific version constraints
		if len(component) == 0 && len(constraints) != 0 {
			v := result[""]
			if v != nil {
				for _, constraintStr := range constraints {
					constraint, err := semver.NewConstraint(constraintStr)
					if err != nil {
						return false
					}

					if !constraint.Check(v) {
						return false
					}
				}
			}
		}

		// check component-specific version constraints
		if len(component) != 0 && len(constraints) != 0 {
			v := result[component]
			if v != nil {
				for _, constraintStr := range constraints {
					constraint, err := semver.NewConstraint(constraintStr)
					if err != nil {
						return false
					}

					if !constraint.Check(v) {
						return false
					}
				}
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
