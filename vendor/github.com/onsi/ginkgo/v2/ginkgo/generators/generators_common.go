package generators

import (
	"go/build"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2/ginkgo/command"
)

type GeneratorsConfig struct {
	Agouti, NoDot, Internal bool
	CustomTemplate          string
}

func getPackageAndFormattedName() (string, string, string) {
	path, err := os.Getwd()
	command.AbortIfError("Could not get current working directory:", err)

	dirName := strings.Replace(filepath.Base(path), "-", "_", -1)
	dirName = strings.Replace(dirName, " ", "_", -1)

	pkg, err := build.ImportDir(path, 0)
	packageName := pkg.Name
	if err != nil {
		packageName = ensureLegalPackageName(dirName)
	}

	formattedName := prettifyName(filepath.Base(path))
	return packageName, dirName, formattedName
}

func ensureLegalPackageName(name string) string {
	if name == "_" {
		return "underscore"
	}
	if len(name) == 0 {
		return "empty"
	}
	n, isDigitErr := strconv.Atoi(string(name[0]))
	if isDigitErr == nil {
		return []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}[n] + name[1:]
	}
	return name
}

func prettifyName(name string) string {
	name = strings.Replace(name, "-", " ", -1)
	name = strings.Replace(name, "_", " ", -1)
	name = strings.Title(name)
	name = strings.Replace(name, " ", "", -1)
	return name
}

func determinePackageName(name string, internal bool) string {
	if internal {
		return name
	}

	return name + "_test"
}
