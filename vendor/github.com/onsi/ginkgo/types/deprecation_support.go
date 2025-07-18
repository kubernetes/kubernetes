package types

import (
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/formatter"
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
		Message: "You are using a custom reporter.  Support for custom reporters will likely be removed in V2.  Most users were using them to generate junit or teamcity reports and this functionality will be merged into the core reporter.  In addition, Ginkgo 2.0 will support emitting a JSON-formatted report that users can then manipulate to generate custom reports.\n\n{{red}}{{bold}}If this change will be impactful to you please leave a comment on {{cyan}}{{underline}}https://github.com/onsi/ginkgo/issues/711{{/}}",
		DocLink: "removed-custom-reporters",
		Version: "1.16.0",
	}
}

func (d deprecations) V1Reporter() Deprecation {
	return Deprecation{
		Message: "You are using a V1 Ginkgo Reporter.  Please update your custom reporter to the new V2 Reporter interface.",
		DocLink: "changed-reporter-interface",
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
		Version: "1.16.5",
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
	out := formatter.F("\n{{light-yellow}}You're using deprecated Ginkgo functionality:{{/}}\n")
	out += formatter.F("{{light-yellow}}============================================={{/}}\n")
	out += formatter.F("{{bold}}{{green}}Ginkgo 2.0{{/}} is under active development and will introduce several new features, improvements, and a small handful of breaking changes.\n")
	out += formatter.F("A release candidate for 2.0 is now available and 2.0 should GA in Fall 2021.  {{bold}}Please give the RC a try and send us feedback!{{/}}\n")
	out += formatter.F("  - To learn more, view the migration guide at {{cyan}}{{underline}}https://github.com/onsi/ginkgo/blob/ver2/docs/MIGRATING_TO_V2.md{{/}}\n")
	out += formatter.F("  - For instructions on using the Release Candidate visit {{cyan}}{{underline}}https://github.com/onsi/ginkgo/blob/ver2/docs/MIGRATING_TO_V2.md#using-the-beta{{/}}\n")
	out += formatter.F("  - To comment, chime in at {{cyan}}{{underline}}https://github.com/onsi/ginkgo/issues/711{{/}}\n\n")

	for deprecation, locations := range d.deprecations {
		out += formatter.Fi(1, "{{yellow}}"+deprecation.Message+"{{/}}\n")
		if deprecation.DocLink != "" {
			out += formatter.Fi(1, "{{bold}}Learn more at:{{/}} {{cyan}}{{underline}}https://github.com/onsi/ginkgo/blob/ver2/docs/MIGRATING_TO_V2.md#%s{{/}}\n", deprecation.DocLink)
		}
		for _, location := range locations {
			out += formatter.Fi(2, "{{gray}}%s{{/}}\n", location)
		}
	}
	out += formatter.F("\n{{gray}}To silence deprecations that can be silenced set the following environment variable:{{/}}\n")
	out += formatter.Fi(1, "{{gray}}ACK_GINKGO_DEPRECATIONS=%s{{/}}\n", config.VERSION)
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
