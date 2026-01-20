// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build aix || dragonfly || freebsd || linux || netbsd || openbsd || solaris || zos

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// osRelease builds a string describing the operating system release based on the
// properties of the os-release file. If no os-release file is found, or if the
// required properties to build the release description string are missing, an empty
// string is returned instead. For more information about os-release files, see:
// https://www.freedesktop.org/software/systemd/man/os-release.html
func osRelease() string {
	file, err := getOSReleaseFile()
	if err != nil {
		return ""
	}

	defer file.Close()

	values := parseOSReleaseFile(file)

	return buildOSRelease(values)
}

// getOSReleaseFile returns a *os.File pointing to one of the well-known os-release
// files, according to their order of preference. If no file can be opened, it
// returns an error.
func getOSReleaseFile() (*os.File, error) {
	return getFirstAvailableFile([]string{"/etc/os-release", "/usr/lib/os-release"})
}

// parseOSReleaseFile process the file pointed by `file` as an os-release file and
// returns a map with the key-values contained in it. Empty lines or lines starting
// with a '#' character are ignored, as well as lines with the missing key=value
// separator. Values are unquoted and unescaped.
func parseOSReleaseFile(file io.Reader) map[string]string {
	values := make(map[string]string)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()

		if skip(line) {
			continue
		}

		key, value, ok := parse(line)
		if ok {
			values[key] = value
		}
	}

	return values
}

// skip reports whether the line is blank or starts with a '#' character, and
// therefore should be skipped from processing.
func skip(line string) bool {
	line = strings.TrimSpace(line)

	return line == "" || strings.HasPrefix(line, "#")
}

// parse attempts to split the provided line on the first '=' character, and then
// sanitize each side of the split before returning them as a key-value pair.
func parse(line string) (string, string, bool) {
	k, v, found := strings.Cut(line, "=")

	if !found || k == "" {
		return "", "", false
	}

	key := strings.TrimSpace(k)
	value := unescape(unquote(strings.TrimSpace(v)))

	return key, value, true
}

// unquote checks whether the string `s` is quoted with double or single quotes
// and, if so, returns a version of the string without them. Otherwise it returns
// the provided string unchanged.
func unquote(s string) string {
	if len(s) < 2 {
		return s
	}

	if (s[0] == '"' || s[0] == '\'') && s[0] == s[len(s)-1] {
		return s[1 : len(s)-1]
	}

	return s
}

// unescape removes the `\` prefix from some characters that are expected
// to have it added in front of them for escaping purposes.
func unescape(s string) string {
	return strings.NewReplacer(
		`\$`, `$`,
		`\"`, `"`,
		`\'`, `'`,
		`\\`, `\`,
		"\\`", "`",
	).Replace(s)
}

// buildOSRelease builds a string describing the OS release based on the properties
// available on the provided map. It favors a combination of the `NAME` and `VERSION`
// properties as first option (falling back to `VERSION_ID` if `VERSION` isn't
// found), and using `PRETTY_NAME` alone if some of the previous are not present. If
// none of these properties are found, it returns an empty string.
//
// The rationale behind not using `PRETTY_NAME` as first choice was that, for some
// Linux distributions, it doesn't include the same detail that can be found on the
// individual `NAME` and `VERSION` properties, and combining `PRETTY_NAME` with
// other properties can produce "pretty" redundant strings in some cases.
func buildOSRelease(values map[string]string) string {
	var osRelease string

	name := values["NAME"]
	version := values["VERSION"]

	if version == "" {
		version = values["VERSION_ID"]
	}

	if name != "" && version != "" {
		osRelease = fmt.Sprintf("%s %s", name, version)
	} else {
		osRelease = values["PRETTY_NAME"]
	}

	return osRelease
}
