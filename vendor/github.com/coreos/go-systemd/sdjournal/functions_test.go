// Copyright 2015 RedHat, Inc.
// Copyright 2015 CoreOS, Inc.
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

package sdjournal

import "testing"

func TestGetFunction(t *testing.T) {
	f, err := getFunction("sd_journal_open")

	if err != nil {
		t.Errorf("Error getting an existing function: %s", err)
	}

	if f == nil {
		t.Error("Got nil function pointer")
	}

	_, err = getFunction("non_existent_function")

	if err == nil {
		t.Error("Expected to get an error, got nil")
	}
}
