/*Package junitxml creates a JUnit XML report from a testjson.Execution.
 */
package junitxml

import (
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"gotest.tools/gotestsum/log"
	"gotest.tools/gotestsum/testjson"
)

// JUnitTestSuites is a collection of JUnit test suites.
type JUnitTestSuites struct {
	XMLName xml.Name `xml:"testsuites"`
	Suites  []JUnitTestSuite
}

// JUnitTestSuite is a single JUnit test suite which may contain many
// testcases.
type JUnitTestSuite struct {
	XMLName    xml.Name        `xml:"testsuite"`
	Tests      int             `xml:"tests,attr"`
	Failures   int             `xml:"failures,attr"`
	Time       string          `xml:"time,attr"`
	Name       string          `xml:"name,attr"`
	Properties []JUnitProperty `xml:"properties>property,omitempty"`
	TestCases  []JUnitTestCase
}

// JUnitTestCase is a single test case with its result.
type JUnitTestCase struct {
	XMLName     xml.Name          `xml:"testcase"`
	Classname   string            `xml:"classname,attr"`
	Name        string            `xml:"name,attr"`
	Time        string            `xml:"time,attr"`
	SkipMessage *JUnitSkipMessage `xml:"skipped,omitempty"`
	Failure     *JUnitFailure     `xml:"failure,omitempty"`
}

// JUnitSkipMessage contains the reason why a testcase was skipped.
type JUnitSkipMessage struct {
	Message string `xml:"message,attr"`
}

// JUnitProperty represents a key/value pair used to define properties.
type JUnitProperty struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

// JUnitFailure contains data related to a failed test.
type JUnitFailure struct {
	Message  string `xml:"message,attr"`
	Type     string `xml:"type,attr"`
	Contents string `xml:",chardata"`
}

// Config used to write a junit XML document.
type Config struct {
	FormatTestSuiteName     FormatFunc
	FormatTestCaseClassname FormatFunc
}

// FormatFunc converts a string from one format into another.
type FormatFunc func(string) string

// Write creates an XML document and writes it to out.
func Write(out io.Writer, exec *testjson.Execution, cfg Config) error {
	if err := write(out, generate(exec, cfg)); err != nil {
		return fmt.Errorf("failed to write JUnit XML: %v", err)
	}
	return nil
}

func generate(exec *testjson.Execution, cfg Config) JUnitTestSuites {
	cfg = configWithDefaults(cfg)
	version := goVersion()
	suites := JUnitTestSuites{}

	for _, pkgname := range exec.Packages() {
		pkg := exec.Package(pkgname)
		junitpkg := JUnitTestSuite{
			Name:       cfg.FormatTestSuiteName(pkgname),
			Tests:      pkg.Total,
			Time:       formatDurationAsSeconds(pkg.Elapsed()),
			Properties: packageProperties(version),
			TestCases:  packageTestCases(pkg, cfg.FormatTestCaseClassname),
			Failures:   len(pkg.Failed),
		}
		suites.Suites = append(suites.Suites, junitpkg)
	}
	return suites
}

func configWithDefaults(cfg Config) Config {
	noop := func(v string) string {
		return v
	}
	if cfg.FormatTestSuiteName == nil {
		cfg.FormatTestSuiteName = noop
	}
	if cfg.FormatTestCaseClassname == nil {
		cfg.FormatTestCaseClassname = noop
	}
	return cfg
}

func formatDurationAsSeconds(d time.Duration) string {
	return fmt.Sprintf("%f", d.Seconds())
}

func packageProperties(goVersion string) []JUnitProperty {
	return []JUnitProperty{
		{Name: "go.version", Value: goVersion},
	}
}

// goVersion returns the version as reported by the go binary in PATH. This
// version will not be the same as runtime.Version, which is always the version
// of go used to build the gotestsum binary.
//
// To skip the os/exec call set the GOVERSION environment variable to the
// desired value.
func goVersion() string {
	if version, ok := os.LookupEnv("GOVERSION"); ok {
		return version
	}
	log.Debugf("exec: go version")
	cmd := exec.Command("go", "version")
	out, err := cmd.Output()
	if err != nil {
		log.Warnf("Failed to lookup go version for junit xml: %v", err)
		return "unknown"
	}
	return strings.TrimPrefix(strings.TrimSpace(string(out)), "go version ")
}

func packageTestCases(pkg *testjson.Package, formatClassname FormatFunc) []JUnitTestCase {
	cases := []JUnitTestCase{}

	if pkg.TestMainFailed() {
		jtc := newJUnitTestCase(testjson.TestCase{Test: "TestMain"}, formatClassname)
		jtc.Failure = &JUnitFailure{
			Message:  "Failed",
			Contents: pkg.Output(0),
		}
		cases = append(cases, jtc)
	}

	for _, tc := range pkg.Failed {
		jtc := newJUnitTestCase(tc, formatClassname)
		jtc.Failure = &JUnitFailure{
			Message:  "Failed",
			Contents: strings.Join(pkg.OutputLines(tc), ""),
		}
		cases = append(cases, jtc)
	}

	for _, tc := range pkg.Skipped {
		jtc := newJUnitTestCase(tc, formatClassname)
		jtc.SkipMessage = &JUnitSkipMessage{
			Message: strings.Join(pkg.OutputLines(tc), ""),
		}
		cases = append(cases, jtc)
	}

	for _, tc := range pkg.Passed {
		jtc := newJUnitTestCase(tc, formatClassname)
		cases = append(cases, jtc)
	}
	return cases
}

func newJUnitTestCase(tc testjson.TestCase, formatClassname FormatFunc) JUnitTestCase {
	return JUnitTestCase{
		Classname: formatClassname(tc.Package),
		Name:      tc.Test.Name(),
		Time:      formatDurationAsSeconds(tc.Elapsed),
	}
}

func write(out io.Writer, suites JUnitTestSuites) error {
	doc, err := xml.MarshalIndent(suites, "", "\t")
	if err != nil {
		return err
	}
	_, err = out.Write([]byte(xml.Header))
	if err != nil {
		return err
	}
	_, err = out.Write(doc)
	return err
}
