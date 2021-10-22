// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rules

import (
	"testing"
)

func TestNameSuffix(t *testing.T) {
	if pass, suggestion := checkNameSuffix("author_name"); !pass {
		t.Error("Given \"author_name\", checkNameSuffix() returned false, expected true")
	} else if suggestion != "author" {
		t.Errorf("Expected suggestion \"author\", received %s instead", suggestion)
	}

	if pass, suggestion := checkNameSuffix("author"); pass {
		t.Error("Given \"author\", checkNameSuffix() returned true, expected false")
	} else if suggestion != "author" {
		t.Errorf("Expected suggestion \"author\", received %s instead", suggestion)
	}

}
