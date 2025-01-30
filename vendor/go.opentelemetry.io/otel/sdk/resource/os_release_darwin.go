// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"encoding/xml"
	"fmt"
	"io"
	"os"
)

type plist struct {
	XMLName xml.Name `xml:"plist"`
	Dict    dict     `xml:"dict"`
}

type dict struct {
	Key    []string `xml:"key"`
	String []string `xml:"string"`
}

// osRelease builds a string describing the operating system release based on the
// contents of the property list (.plist) system files. If no .plist files are found,
// or if the required properties to build the release description string are missing,
// an empty string is returned instead. The generated string resembles the output of
// the `sw_vers` commandline program, but in a single-line string. For more information
// about the `sw_vers` program, see: https://www.unix.com/man-page/osx/1/SW_VERS.
func osRelease() string {
	file, err := getPlistFile()
	if err != nil {
		return ""
	}

	defer file.Close()

	values, err := parsePlistFile(file)
	if err != nil {
		return ""
	}

	return buildOSRelease(values)
}

// getPlistFile returns a *os.File pointing to one of the well-known .plist files
// available on macOS. If no file can be opened, it returns an error.
func getPlistFile() (*os.File, error) {
	return getFirstAvailableFile([]string{
		"/System/Library/CoreServices/SystemVersion.plist",
		"/System/Library/CoreServices/ServerVersion.plist",
	})
}

// parsePlistFile process the file pointed by `file` as a .plist file and returns
// a map with the key-values for each pair of correlated <key> and <string> elements
// contained in it.
func parsePlistFile(file io.Reader) (map[string]string, error) {
	var v plist

	err := xml.NewDecoder(file).Decode(&v)
	if err != nil {
		return nil, err
	}

	if len(v.Dict.Key) != len(v.Dict.String) {
		return nil, fmt.Errorf("the number of <key> and <string> elements doesn't match")
	}

	properties := make(map[string]string, len(v.Dict.Key))
	for i, key := range v.Dict.Key {
		properties[key] = v.Dict.String[i]
	}

	return properties, nil
}

// buildOSRelease builds a string describing the OS release based on the properties
// available on the provided map. It tries to find the `ProductName`, `ProductVersion`
// and `ProductBuildVersion` properties. If some of these properties are not found,
// it returns an empty string.
func buildOSRelease(properties map[string]string) string {
	productName := properties["ProductName"]
	productVersion := properties["ProductVersion"]
	productBuildVersion := properties["ProductBuildVersion"]

	if productName == "" || productVersion == "" || productBuildVersion == "" {
		return ""
	}

	return fmt.Sprintf("%s %s (%s)", productName, productVersion, productBuildVersion)
}
