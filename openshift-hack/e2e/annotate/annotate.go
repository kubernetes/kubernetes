package annotate

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
)

var reHasSig = regexp.MustCompile(`\[sig-[\w-]+\]`)

// Run generates tests annotations for the targeted package.
// It accepts testMaps which defines labeling rules and filter
// function to remove elements based on test name and their labels.
func Run(testMaps map[string][]string, filter func(name string) bool) {
	var errors []string

	if len(os.Args) != 2 && len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "error: requires exactly one argument\n")
		os.Exit(1)
	}
	filename := os.Args[len(os.Args)-1]

	generator := newGenerator(testMaps)
	ginkgo.GetSuite().BuildTree()
	ginkgo.GetSuite().WalkTests(generator.generateRename)
	if len(generator.errors) > 0 {
		errors = append(errors, generator.errors...)
	}

	renamer := newRenamerFromGenerated(generator.output)
	// generated file has a map[string]string in the following format:
	// original k8s name: k8s name with our labels at the end
	ginkgo.GetSuite().WalkTests(renamer.updateNodeText)
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
	for testName, labels := range generator.output {
		if filter(fmt.Sprintf("%s%s", testName, labels)) {
			continue
		}
		pairs = append(pairs, fmt.Sprintf("%q:\n%q,", testName, labels))
	}
	sort.Strings(pairs)
	contents := fmt.Sprintf(`
package generated

import (
	"fmt"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
)

var Annotations = map[string]string{
%s
}

func init() {
	ginkgo.GetSuite().SetAnnotateFn(func(name string, node types.TestSpec) {
		if newLabels, ok := Annotations[name]; ok {
			node.AppendText(newLabels)
		} else {
			panic(fmt.Sprintf("unable to find test %%s", name))
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

func newGenerator(testMaps map[string][]string) *ginkgoTestRenamer {
	var allLabels []string
	matches := make(map[string]*regexp.Regexp)
	stringMatches := make(map[string][]string)

	for label, items := range testMaps {
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
	sort.Strings(allLabels)

	excludedTestsFilter := regexp.MustCompile(strings.Join(ExcludedTests, `|`))

	return &ginkgoTestRenamer{
		allLabels:           allLabels,
		stringMatches:       stringMatches,
		matches:             matches,
		excludedTestsFilter: excludedTestsFilter,
		output:              make(map[string]string),
	}
}

func newRenamerFromGenerated(names map[string]string) *ginkgoTestRenamer {
	return &ginkgoTestRenamer{
		output:  names,
		missing: make(map[string]struct{}),
	}
}

type ginkgoTestRenamer struct {
	// keys defined in TestMaps in openshift-hack/e2e/annotate/rules.go
	allLabels []string
	// exact substrings to match to apply a particular label
	stringMatches map[string][]string
	// regular expressions to match to apply a particular label
	matches map[string]*regexp.Regexp
	// regular expression excluding permanently a set of tests
	// see ExcludedTests in openshift-hack/e2e/annotate/rules.go
	excludedTestsFilter *regexp.Regexp

	// output from the generateRename and also input for updateNodeText
	output map[string]string
	// map of unmatched test names
	missing map[string]struct{}
	// a list of errors to display
	errors []string
}

func (r *ginkgoTestRenamer) updateNodeText(name string, node types.TestSpec) {
	if newLables, ok := r.output[name]; ok {
		node.AppendText(newLables)
	} else {
		r.missing[name] = struct{}{}
	}
}

func (r *ginkgoTestRenamer) generateRename(name string, node types.TestSpec) {
	newLabels := ""
	newName := name
	for {
		count := 0
		for _, label := range r.allLabels {
			// never apply a sig label twice
			if strings.HasPrefix(label, "[sig-") && strings.Contains(newName, "[sig-") {
				continue
			}
			if strings.Contains(newName, label) {
				continue
			}

			var hasLabel bool
			for _, segment := range r.stringMatches[label] {
				hasLabel = strings.Contains(newName, segment)
				if hasLabel {
					break
				}
			}
			if !hasLabel {
				if re := r.matches[label]; re != nil {
					hasLabel = r.matches[label].MatchString(newName)
				}
			}

			if hasLabel {
				count++
				newLabels += " " + label
				newName += " " + label
			}
		}
		if count == 0 {
			break
		}
	}

	// Append suite name to test, if it doesn't already have one
	if !r.excludedTestsFilter.MatchString(newName) && !strings.Contains(newName, "[Suite:") {
		isSerial := strings.Contains(newName, "[Serial]")
		isConformance := strings.Contains(newName, "[Conformance]")
		switch {
		case isSerial && isConformance:
			newLabels += " [Suite:openshift/conformance/serial/minimal]"
		case isSerial:
			newLabels += " [Suite:openshift/conformance/serial]"
		case isConformance:
			newLabels += " [Suite:openshift/conformance/parallel/minimal]"
		default:
			newLabels += " [Suite:openshift/conformance/parallel]"
		}
	}
	codeLocations := node.CodeLocations()
	if isGoModulePath(codeLocations[len(codeLocations)-1].FileName, "k8s.io/kubernetes", "test/e2e") {
		newLabels += " [Suite:k8s]"
	}

	if err := checkBalancedBrackets(newName); err != nil {
		r.errors = append(r.errors, err.Error())
	}
	r.output[name] = newLabels
}

// isGoModulePath returns true if the packagePath reported by reflection is within a
// module and given module path. When go mod is in use, module and modulePath are not
// contiguous as they were in older golang versions with vendoring, so naive contains
// tests fail.
//
// historically: ".../vendor/k8s.io/kubernetes/test/e2e"
// go.mod:       "k8s.io/kubernetes@0.18.4/test/e2e"
func isGoModulePath(packagePath, module, modulePath string) bool {
	return regexp.MustCompile(fmt.Sprintf(`\b%s(@[^/]*|)/%s\b`, regexp.QuoteMeta(module), regexp.QuoteMeta(modulePath))).MatchString(packagePath)
}

// checkBalancedBrackets ensures that square brackets are balanced in generated test
// names. If they are not, it returns an error with the name of the test and a guess
// where the unmatched bracket(s) are.
func checkBalancedBrackets(testName string) error {
	stack := make([]int, 0, len(testName))
	for idx, c := range testName {
		if c == '[' {
			stack = append(stack, idx)
		} else if c == ']' {
			// case when we start off with a ]
			if len(stack) == 0 {
				stack = append(stack, idx)
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}

	if len(stack) > 0 {
		msg := testName + "\n"
	outerLoop:
		for i := 0; i < len(testName); i++ {
			for _, loc := range stack {
				if i == loc {
					msg += "^"
					continue outerLoop
				}
			}
			msg += " "
		}
		return fmt.Errorf("unbalanced brackets in test name:\n%s\n", msg)
	}

	return nil
}
