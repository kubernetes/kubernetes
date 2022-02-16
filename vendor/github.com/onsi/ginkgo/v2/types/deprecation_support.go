package types

import (
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/onsi/ginkgo/v2/formatter"
)

type Deprecation struct {
	Message string
	DocLink string
	Version string
}

type deprecations struct{}

var Deprecations = deprecations{}

func (d deprecations) CustomReporter() Deprecation {
	return Deprecation{
		Message: "Support for custom reporters has been removed in V2.  Please read the documentation linked to below for Ginkgo's new behavior and for a migration path:",
		DocLink: "removed-custom-reporters",
		Version: "1.16.0",
	}
}

func (d deprecations) Async() Deprecation {
	return Deprecation{
		Message: "You are passing a Done channel to a test node to test asynchronous behavior.  This is deprecated in Ginkgo V2.  Your test will run synchronously and the timeout will be ignored.",
		DocLink: "removed-async-testing",
		Version: "1.16.0",
	}
}

func (d deprecations) Measure() Deprecation {
	return Deprecation{
		Message: "Measure is deprecated and will be removed in Ginkgo V2.  Please migrate to gomega/gmeasure.",
		DocLink: "removed-measure",
		Version: "1.16.3",
	}
}

func (d deprecations) ParallelNode() Deprecation {
	return Deprecation{
		Message: "GinkgoParallelNode is deprecated and will be removed in Ginkgo V2.  Please use GinkgoParallelProcess instead.",
		DocLink: "renamed-ginkgoparallelnode",
		Version: "1.16.4",
	}
}

func (d deprecations) CurrentGinkgoTestDescription() Deprecation {
	return Deprecation{
		Message: "CurrentGinkgoTestDescription() is deprecated in Ginkgo V2.  Use CurrentSpecReport() instead.",
		DocLink: "changed-currentginkgotestdescription",
		Version: "1.16.0",
	}
}

func (d deprecations) Convert() Deprecation {
	return Deprecation{
		Message: "The convert command is deprecated in Ginkgo V2",
		DocLink: "removed-ginkgo-convert",
		Version: "1.16.0",
	}
}

func (d deprecations) Blur() Deprecation {
	return Deprecation{
		Message: "The blur command is deprecated in Ginkgo V2.  Use 'ginkgo unfocus' instead.",
		Version: "1.16.0",
	}
}

func (d deprecations) Nodot() Deprecation {
	return Deprecation{
		Message: "The nodot command is deprecated in Ginkgo V2.  Please either dot-import Ginkgo or use the package identifier in your code to references objects and types provided by Ginkgo and Gomega.",
		DocLink: "removed-ginkgo-nodot",
		Version: "1.16.0",
	}
}

type DeprecationTracker struct {
	deprecations map[Deprecation][]CodeLocation
}

func NewDeprecationTracker() *DeprecationTracker {
	return &DeprecationTracker{
		deprecations: map[Deprecation][]CodeLocation{},
	}
}

func (d *DeprecationTracker) TrackDeprecation(deprecation Deprecation, cl ...CodeLocation) {
	ackVersion := os.Getenv("ACK_GINKGO_DEPRECATIONS")
	if deprecation.Version != "" && ackVersion != "" {
		ack := ParseSemVer(ackVersion)
		version := ParseSemVer(deprecation.Version)
		if ack.GreaterThanOrEqualTo(version) {
			return
		}
	}

	if len(cl) == 1 {
		d.deprecations[deprecation] = append(d.deprecations[deprecation], cl[0])
	} else {
		d.deprecations[deprecation] = []CodeLocation{}
	}
}

func (d *DeprecationTracker) DidTrackDeprecations() bool {
	return len(d.deprecations) > 0
}

func (d *DeprecationTracker) DeprecationsReport() string {
	out := formatter.F("{{light-yellow}}You're using deprecated Ginkgo functionality:{{/}}\n")
	out += formatter.F("{{light-yellow}}============================================={{/}}\n")
	for deprecation, locations := range d.deprecations {
		out += formatter.Fi(1, "{{yellow}}"+deprecation.Message+"{{/}}\n")
		if deprecation.DocLink != "" {
			out += formatter.Fi(1, "{{bold}}Learn more at:{{/}} {{cyan}}{{underline}}https://onsi.github.io/ginkgo/MIGRATING_TO_V2#%s{{/}}\n", deprecation.DocLink)
		}
		for _, location := range locations {
			out += formatter.Fi(2, "{{gray}}%s{{/}}\n", location)
		}
	}
	out += formatter.F("\n{{gray}}To silence deprecations that can be silenced set the following environment variable:{{/}}\n")
	out += formatter.Fi(1, "{{gray}}ACK_GINKGO_DEPRECATIONS=%s{{/}}\n", VERSION)
	return out
}

type SemVer struct {
	Major int
	Minor int
	Patch int
}

func (s SemVer) GreaterThanOrEqualTo(o SemVer) bool {
	return (s.Major > o.Major) ||
		(s.Major == o.Major && s.Minor > o.Minor) ||
		(s.Major == o.Major && s.Minor == o.Minor && s.Patch >= o.Patch)
}

func ParseSemVer(semver string) SemVer {
	out := SemVer{}
	semver = strings.TrimFunc(semver, func(r rune) bool {
		return !(unicode.IsNumber(r) || r == '.')
	})
	components := strings.Split(semver, ".")
	if len(components) > 0 {
		out.Major, _ = strconv.Atoi(components[0])
	}
	if len(components) > 1 {
		out.Minor, _ = strconv.Atoi(components[1])
	}
	if len(components) > 2 {
		out.Patch, _ = strconv.Atoi(components[2])
	}
	return out
}
