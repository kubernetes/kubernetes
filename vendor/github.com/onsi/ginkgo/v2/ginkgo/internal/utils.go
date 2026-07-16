package internal

import (
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/ginkgo/command"
)

func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func CopyFile(src string, dest string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}

	srcStat, err := srcFile.Stat()
	if err != nil {
		return err
	}

	if _, err := os.Stat(dest); err == nil {
		os.Remove(dest)
	}

	destFile, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE, srcStat.Mode())
	if err != nil {
		return err
	}

	_, err = io.Copy(destFile, srcFile)
	if err != nil {
		return err
	}

	if err := srcFile.Close(); err != nil {
		return err
	}
	return destFile.Close()
}

func GoFmt(path string) {
	out, err := exec.Command("go", "fmt", path).CombinedOutput()
	if err != nil {
		command.AbortIfError(fmt.Sprintf("Could not fmt:\n%s\n", string(out)), err)
	}
}

func PluralizedWord(singular, plural string, count int) string {
	if count == 1 {
		return singular
	}
	return plural
}

func FailedSuitesReport(suites TestSuites, f formatter.Formatter) string {
	out := ""
	out += "There were failures detected in the following suites:\n"

	maxPackageNameLength := 0
	for _, suite := range suites.WithState(TestSuiteStateFailureStates...) {
		if len(suite.PackageName) > maxPackageNameLength {
			maxPackageNameLength = len(suite.PackageName)
		}
	}

	packageNameFormatter := fmt.Sprintf("%%%ds", maxPackageNameLength)
	for _, suite := range suites {
		switch suite.State {
		case TestSuiteStateFailed:
			out += f.Fi(1, "{{red}}"+packageNameFormatter+" {{gray}}%s{{/}}\n", suite.PackageName, suite.Path)
		case TestSuiteStateFailedToCompile:
			out += f.Fi(1, "{{red}}"+packageNameFormatter+" {{gray}}%s {{magenta}}[Compilation failure]{{/}}\n", suite.PackageName, suite.Path)
		case TestSuiteStateFailedDueToTimeout:
			out += f.Fi(1, "{{red}}"+packageNameFormatter+" {{gray}}%s {{orange}}[%s]{{/}}\n", suite.PackageName, suite.Path, TIMEOUT_ELAPSED_FAILURE_REASON)
		}
	}
	return out
}
