package internal

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/types"
)

var versiorRe = regexp.MustCompile(`v(\d+\.\d+\.\d+)`)

func VerifyCLIAndFrameworkVersion(suites TestSuites) {
	cliVersion := types.VERSION
	mismatches := map[string][]string{}

	for _, suite := range suites {
		cmd := exec.Command("go", "list", "-m", "github.com/onsi/ginkgo/v2")
		cmd.Dir = suite.Path
		output, err := cmd.CombinedOutput()
		if err != nil {
			continue
		}
		components := strings.Split(string(output), " ")
		if len(components) != 2 {
			continue
		}
		matches := versiorRe.FindStringSubmatch(components[1])
		if matches == nil || len(matches) != 2 {
			continue
		}
		libraryVersion := matches[1]
		if cliVersion != libraryVersion {
			mismatches[libraryVersion] = append(mismatches[libraryVersion], suite.PackageName)
		}
	}

	if len(mismatches) == 0 {
		return
	}

	fmt.Println(formatter.F("{{red}}{{bold}}Ginkgo detected a version mismatch between the Ginkgo CLI and the version of Ginkgo imported by your packages:{{/}}"))

	fmt.Println(formatter.Fi(1, "Ginkgo CLI Version:"))
	fmt.Println(formatter.Fi(2, "{{bold}}%s{{/}}", cliVersion))
	fmt.Println(formatter.Fi(1, "Mismatched package versions found:"))
	for version, packages := range mismatches {
		fmt.Println(formatter.Fi(2, "{{bold}}%s{{/}} used by %s", version, strings.Join(packages, ", ")))
	}
	fmt.Println("")
	fmt.Println(formatter.Fiw(1, formatter.COLS, "{{gray}}Ginkgo will continue to attempt to run but you may see errors (including flag parsing errors) and should either update your go.mod or your version of the Ginkgo CLI to match.\n\nTo install the matching version of the CLI run\n  {{bold}}go install github.com/onsi/ginkgo/v2/ginkgo{{/}}{{gray}}\nfrom a path that contains a go.mod file.  Alternatively you can use\n  {{bold}}go run github.com/onsi/ginkgo/v2/ginkgo{{/}}{{gray}}\nfrom a path that contains a go.mod file to invoke the matching version of the Ginkgo CLI.\n\nIf you are attempting to test multiple packages that each have a different version of the Ginkgo library with a single Ginkgo CLI that is currently unsupported.\n{{/}}"))
}
