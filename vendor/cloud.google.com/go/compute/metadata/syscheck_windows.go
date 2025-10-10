// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build windows

package metadata

import (
	"strings"

	"golang.org/x/sys/windows/registry"
)

// NOTE: systemInfoSuggestsGCE is assigned to a varible for test stubbing purposes.
var systemInfoSuggestsGCE = func() bool {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `SYSTEM\HardwareConfig\Current`, registry.QUERY_VALUE)
	if err != nil {
		return false
	}
	defer k.Close()

	s, _, err := k.GetStringValue("SystemProductName")
	if err != nil {
		return false
	}
	s = strings.TrimSpace(s)
	return strings.HasPrefix(s, "Google")
}
