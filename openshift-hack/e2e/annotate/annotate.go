package annotate

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"

	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/types"
)

var reHasSig = regexp.MustCompile(`\[sig-[\w-]+\]`)

// Run generates tests annotations for the targeted package.
func Run() {
	if len(os.Args) != 2 && len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "error: requires exactly one argument\n")
		os.Exit(1)
	}
	filename := os.Args[len(os.Args)-1]

	generator := newGenerator()
	ginkgo.WalkTests(generator.generateRename)

	renamer := newRenamerFromGenerated(generator.output)
	ginkgo.WalkTests(renamer.updateNodeText)
	if len(renamer.missing) > 0 {
		var names []string
		for name := range renamer.missing {
			names = append(names, name)
		}
		sort.Strings(names)
		fmt.Fprintf(os.Stderr, "failed:\n%s\n", strings.Join(names, "\n"))
		os.Exit(1)
	}

	// All tests must be associated with a sig (either upstream), or downstream
	// If you get this error, you should add the [sig-X] tag to your test (if its
	// in origin) or if it is upstream add a new rule to rules.go that assigns
	// the test in question to the right sig.
	//
	// Upstream sigs map to teams (if you have representation on that sig, you
	//   own those tests in origin)
	// Downstream sigs: sig-imageregistry, sig-builds, sig-devex
	var errors []string
	for from, to := range generator.output {
		if !reHasSig.MatchString(from) && !reHasSig.MatchString(to) {
			errors = append(errors, fmt.Sprintf("all tests must define a [sig-XXXX] tag or have a rule %q", from))
		}
	}
	if len(errors) > 0 {
		sort.Strings(errors)
		for _, s := range errors {
			fmt.Fprintf(os.Stderr, "failed: %s\n", s)
		}
		os.Exit(1)
	}

	var pairs []string
	for from, to := range generator.output {
		pairs = append(pairs, fmt.Sprintf("%q:\n%q,", from, to))
	}
	sort.Strings(pairs)
	contents := fmt.Sprintf(`
package generated

import (
	"fmt"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/types"
)

var annotations = map[string]string{
%s
}

func init() {
	ginkgo.WalkTests(func(name, parentName string, node types.TestNode) {
		combined := name
		if len(parentName) > 0 {
			combined = parentName + " " + combined
		}
		if updated, ok := annotations[combined]; ok {
			node.SetText(updated)
		} else {
			panic(fmt.Sprintf("unable to find test %%s", combined))
		}
	})
}
`, strings.Join(pairs, "\n\n"))
	if err := ioutil.WriteFile(filename, []byte(contents), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v", err)
		os.Exit(1)
	}
	if _, err := exec.Command("gofmt", "-s", "-w", filename).Output(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v", err)
		os.Exit(1)
	}
}

func newGenerator() *ginkgoTestRenamer {
	var allLabels []string
	matches := make(map[string]*regexp.Regexp)
	stringMatches := make(map[string][]string)
	excludes := make(map[string]*regexp.Regexp)

	for label, items := range TestMaps {
		sort.Strings(items)
		allLabels = append(allLabels, label)
		var remain []string
		for _, item := range items {
			re := regexp.MustCompile(item)
			if p, ok := re.LiteralPrefix(); ok {
				stringMatches[label] = append(stringMatches[label], p)
			} else {
				remain = append(remain, item)
			}
		}
		if len(remain) > 0 {
			matches[label] = regexp.MustCompile(strings.Join(remain, `|`))
		}
	}
	for label, items := range LabelExcludes {
		sort.Strings(items)
		excludes[label] = regexp.MustCompile(strings.Join(items, `|`))
	}
	sort.Strings(allLabels)

	excludedTestsFilter := regexp.MustCompile(strings.Join(ExcludedTests, `|`))

	return &ginkgoTestRenamer{
		allLabels:           allLabels,
		stringMatches:       stringMatches,
		matches:             matches,
		excludes:            excludes,
		excludedTestsFilter: excludedTestsFilter,

		output: make(map[string]string),
	}
}

func newRenamerFromGenerated(names map[string]string) *ginkgoTestRenamer {
	return &ginkgoTestRenamer{
		output:  names,
		missing: make(map[string]struct{}),
	}
}

type ginkgoTestRenamer struct {
	allLabels           []string
	stringMatches       map[string][]string
	matches             map[string]*regexp.Regexp
	excludes            map[string]*regexp.Regexp
	excludedTestsFilter *regexp.Regexp

	output  map[string]string
	missing map[string]struct{}
}

func (r *ginkgoTestRenamer) updateNodeText(name, parentName string, node types.TestNode) {
	if updated, ok := r.output[combineNames(parentName, name)]; ok {
		node.SetText(updated)
	} else {
		r.missing[combineNames(parentName, name)] = struct{}{}
	}
}

func (r *ginkgoTestRenamer) generateRename(name, parentName string, node types.TestNode) {
	originalName := name
	combinedName := combineNames(parentName, name)

	labels := ""
	for {
		count := 0
		for _, label := range r.allLabels {
			// never apply a sig label twice
			if strings.HasPrefix(label, "[sig-") && strings.Contains(combinedName, "[sig-") {
				continue
			}
			if strings.Contains(combinedName, label) {
				continue
			}

			var hasLabel bool
			for _, segment := range r.stringMatches[label] {
				hasLabel = strings.Contains(combinedName, segment)
				if hasLabel {
					break
				}
			}
			if !hasLabel {
				if re := r.matches[label]; re != nil {
					hasLabel = r.matches[label].MatchString(combinedName)
				}
			}

			if hasLabel {
				// TODO: remove when we no longer need it
				if re, ok := r.excludes[label]; ok && re.MatchString(combinedName) {
					continue
				}
				count++
				labels += " " + label
				combinedName += " " + label
				name += " " + label
			}
		}
		if count == 0 {
			break
		}
	}

	if !r.excludedTestsFilter.MatchString(combinedName) {
		isSerial := strings.Contains(combinedName, "[Serial]")
		isConformance := strings.Contains(combinedName, "[Conformance]")
		switch {
		case isSerial && isConformance:
			name += " [Suite:openshift/conformance/serial/minimal]"
		case isSerial:
			name += " [Suite:openshift/conformance/serial]"
		case isConformance:
			name += " [Suite:openshift/conformance/parallel/minimal]"
		default:
			name += " [Suite:openshift/conformance/parallel]"
		}
	}
	if isGoModulePath(node.CodeLocation().FileName, "k8s.io/kubernetes", "test/e2e") {
		name += " [Suite:k8s]"
	}

	r.output[combineNames(parentName, originalName)] = name
}

// isGoModulePath returns true if the packagePath reported by reflection is within a
// module and given module path. When go mod is in use, module and modulePath are not
// contiguous as they were in older golang versions with vendoring, so naive contains
// tests fail.
//
// historically: ".../vendor/k8s.io/kubernetes/test/e2e"
// go.mod:       "k8s.io/kubernetes@0.18.4/test/e2e"
//
func isGoModulePath(packagePath, module, modulePath string) bool {
	return regexp.MustCompile(fmt.Sprintf(`\b%s(@[^/]*|)/%s\b`, regexp.QuoteMeta(module), regexp.QuoteMeta(modulePath))).MatchString(packagePath)
}

func combineNames(parentName, name string) string {
	if len(parentName) == 0 {
		return name
	}
	return parentName + " " + name
}
