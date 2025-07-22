// Copyright 2020 Google Inc. All Rights Reserved.
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

package machine

import (
	"fmt"

	"golang.org/x/sys/windows/registry"
)

func getOperatingSystem() (string, error) {
	system := "Windows"
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion`, registry.QUERY_VALUE)
	if err != nil {
		return system, err
	}
	defer k.Close()

	productName, _, err := k.GetStringValue("ProductName")
	if err != nil {
		return system, nil
	}

	releaseId, _, err := k.GetStringValue("ReleaseId")
	if err != nil {
		return system, err
	}

	currentBuildNumber, _, err := k.GetStringValue("CurrentBuildNumber")
	if err != nil {
		return system, err
	}
	revision, _, err := k.GetIntegerValue("UBR")
	if err != nil {
		return system, err
	}

	system = fmt.Sprintf("%s Version %s (OS Build %s.%d)",
		productName, releaseId, currentBuildNumber, revision)

	return system, nil
}
