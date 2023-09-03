// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"fmt"
	"strconv"

	"golang.org/x/sys/windows/registry"
)

// platformOSDescription returns a human readable OS version information string.
// It does so by querying registry values under the
// `SOFTWARE\Microsoft\Windows NT\CurrentVersion` key. The final string
// resembles the one displayed by the Version Reporter Applet (winver.exe).
func platformOSDescription() (string, error) {
	k, err := registry.OpenKey(
		registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion`, registry.QUERY_VALUE)

	if err != nil {
		return "", err
	}

	defer k.Close()

	var (
		productName               = readProductName(k)
		displayVersion            = readDisplayVersion(k)
		releaseID                 = readReleaseID(k)
		currentMajorVersionNumber = readCurrentMajorVersionNumber(k)
		currentMinorVersionNumber = readCurrentMinorVersionNumber(k)
		currentBuildNumber        = readCurrentBuildNumber(k)
		ubr                       = readUBR(k)
	)

	if displayVersion != "" {
		displayVersion += " "
	}

	return fmt.Sprintf("%s %s(%s) [Version %s.%s.%s.%s]",
		productName,
		displayVersion,
		releaseID,
		currentMajorVersionNumber,
		currentMinorVersionNumber,
		currentBuildNumber,
		ubr,
	), nil
}

func getStringValue(name string, k registry.Key) string {
	value, _, _ := k.GetStringValue(name)

	return value
}

func getIntegerValue(name string, k registry.Key) uint64 {
	value, _, _ := k.GetIntegerValue(name)

	return value
}

func readProductName(k registry.Key) string {
	return getStringValue("ProductName", k)
}

func readDisplayVersion(k registry.Key) string {
	return getStringValue("DisplayVersion", k)
}

func readReleaseID(k registry.Key) string {
	return getStringValue("ReleaseID", k)
}

func readCurrentMajorVersionNumber(k registry.Key) string {
	return strconv.FormatUint(getIntegerValue("CurrentMajorVersionNumber", k), 10)
}

func readCurrentMinorVersionNumber(k registry.Key) string {
	return strconv.FormatUint(getIntegerValue("CurrentMinorVersionNumber", k), 10)
}

func readCurrentBuildNumber(k registry.Key) string {
	return getStringValue("CurrentBuildNumber", k)
}

func readUBR(k registry.Key) string {
	return strconv.FormatUint(getIntegerValue("UBR", k), 10)
}
