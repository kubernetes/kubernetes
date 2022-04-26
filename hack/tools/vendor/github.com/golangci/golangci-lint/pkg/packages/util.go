package packages

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/tools/go/packages"
)

// reFile matches a line who starts with path and position.
// ex: `/example/main.go:11:17: foobar`
var reFile = regexp.MustCompile(`^.+\.go:\d+:\d+: .+`)

func ExtractErrors(pkg *packages.Package) []packages.Error {
	errors := extractErrorsImpl(pkg, map[*packages.Package]bool{})
	if len(errors) == 0 {
		return errors
	}

	seenErrors := map[string]bool{}
	var uniqErrors []packages.Error
	for _, err := range errors {
		msg := stackCrusher(err.Error())
		if seenErrors[msg] {
			continue
		}

		if msg != err.Error() {
			continue
		}

		seenErrors[msg] = true

		uniqErrors = append(uniqErrors, err)
	}

	if len(pkg.GoFiles) != 0 {
		// errors were extracted from deps and have at least one file in package
		for i := range uniqErrors {
			if _, parseErr := ParseErrorPosition(uniqErrors[i].Pos); parseErr == nil {
				continue
			}

			// change pos to local file to properly process it by processors (properly read line etc.)
			uniqErrors[i].Msg = fmt.Sprintf("%s: %s", uniqErrors[i].Pos, uniqErrors[i].Msg)
			uniqErrors[i].Pos = fmt.Sprintf("%s:1", pkg.GoFiles[0])
		}

		// some errors like "code in directory  expects import" don't have Pos, set it here
		for i := range uniqErrors {
			err := &uniqErrors[i]
			if err.Pos == "" {
				err.Pos = fmt.Sprintf("%s:1", pkg.GoFiles[0])
			}
		}
	}

	return uniqErrors
}

func extractErrorsImpl(pkg *packages.Package, seenPackages map[*packages.Package]bool) []packages.Error {
	if seenPackages[pkg] {
		return nil
	}
	seenPackages[pkg] = true

	if !pkg.IllTyped { // otherwise, it may take hours to traverse all deps many times
		return nil
	}

	if len(pkg.Errors) > 0 {
		return pkg.Errors
	}

	var errors []packages.Error
	for _, iPkg := range pkg.Imports {
		iPkgErrors := extractErrorsImpl(iPkg, seenPackages)
		if iPkgErrors != nil {
			errors = append(errors, iPkgErrors...)
		}
	}

	return errors
}

func stackCrusher(msg string) string {
	index := strings.Index(msg, "(")
	lastIndex := strings.LastIndex(msg, ")")

	if index == -1 || index == len(msg)-1 || lastIndex == -1 || lastIndex != len(msg)-1 {
		return msg
	}

	frag := msg[index+1 : lastIndex]

	if !reFile.MatchString(frag) {
		return msg
	}

	return stackCrusher(frag)
}
