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

func TestSnakeCase(t *testing.T) {
	if pass, suggestion := checkSnakeCase("helloWorld"); pass {
		t.Error("Given \"helloWorld\", snakeCase() returned true, expected false")
	} else if suggestion != "hello_world" {
		t.Errorf("Expected suggestion \"hello_world\", received %s instead", suggestion)
	}

	if pass, suggestion := checkSnakeCase("hello_world"); !pass {
		t.Error("Given \"hello_world\", snakeCase() returned false, expected true")
	} else if suggestion != "hello_world" {
		t.Errorf("Expected suggestion \"hello_world\", received %s instead", suggestion)
	}

}

func TestAbbreviation(t *testing.T) {
	if pass, suggestion := checkAbbreviation("configuration"); !pass {
		t.Error("Given \"configuration\", checkAbbreviation() returned false, expected true")
	} else if suggestion != "config" {
		t.Errorf("Expected suggestion \"config\", received %s instead", suggestion)
	}

	if pass, suggestion := checkAbbreviation("identifier"); !pass {
		t.Error("Given \"identifier\", checkAbbreviation() returned false, expected true")
	} else if suggestion != "id" {
		t.Errorf("Expected suggestion \"id\", received %s instead", suggestion)
	}

	if pass, suggestion := checkAbbreviation("information"); !pass {
		t.Error("Given \"information\", checkAbbreviation() returned false, expected true")
	} else if suggestion != "info" {
		t.Errorf("Expected suggestion \"info\", received %s instead", suggestion)
	}

	if pass, suggestion := checkAbbreviation("specification"); !pass {
		t.Error("Given \"specification\", checkAbbreviation() returned false, expected true")
	} else if suggestion != "spec" {
		t.Errorf("Expected suggestion \"spec\", received %s instead", suggestion)
	}

	if pass, suggestion := checkAbbreviation("statistics"); !pass {
		t.Error("Given \"statistics\", checkAbbreviation() returned false, expected true")
	} else if suggestion != "stats" {
		t.Errorf("Expected suggestion \"stats\", received %s instead", suggestion)
	}

	if pass, suggestion := checkAbbreviation("supercalifrag"); pass {
		t.Error("Given \"supercalifrag\", checkAbbreviation() returned true, expected false")
	} else if suggestion != "supercalifrag" {
		t.Errorf("Expected suggestion \"superalifrag\", received %s instead", suggestion)
	}

}

func TestNumbers(t *testing.T) {
	if pass := checkNumbers("90th_percentile"); !pass {
		t.Error("Given \"90th_percentile\", checkNumbers() returned false, expected true")
	}

	if pass := checkNumbers("hello_2nd_world"); !pass {
		t.Error("Given \"hello_2nd_world\", checkNumbers() returned false, expected true")
	}
	if pass := checkNumbers("second"); pass {
		t.Error("Given \"second\", checkNumbers() returned true, expected false")
	}

}

func TestReservedWords(t *testing.T) {
	if pass := checkReservedWords("catch"); !pass {
		t.Error("Given \"catch\", numbers() returned false, expected true")
	}

	if pass := checkReservedWords("all_except"); !pass {
		t.Error("Given \"all_except\", checkReservedWords() returned false, expected true")
	}

	if pass := checkReservedWords("export"); !pass {
		t.Error("Given \"export\", checkReservedWords() returned false, expected true")
	}

	if pass := checkReservedWords("interface"); !pass {
		t.Error("Given \"interface\", checkReservedWords() returned false, expected true")
	}

	if pass := checkReservedWords("magic"); pass {
		t.Error("Given \"magic\", checkReservedWords() returned true, expected false")
	}

}

func TestPrepositions(t *testing.T) {
	if pass := checkPrepositions("written_by"); !pass {
		t.Error("Given \"written_by\", checkPrepositions() returned false, expected true")
	}

	if pass := checkPrepositions("all_except"); !pass {
		t.Error("Given \"all_except\", checkPrepositions() returned false, expected true")
	}

	if pass := checkPrepositions("process_after"); !pass {
		t.Error("Given \"process_after\", checkPrepositions() returned false, expected true")
	}

	if pass := checkPrepositions("between_rocks_by_shore"); !pass {
		t.Error("Given \"between_rocks_by_shore\", checkPrepositions() returned false, expected true")
	}

	if pass := checkPrepositions("no_preps_here"); pass {
		t.Error("Given \"magic\", checkPrepositions() returned true, expected false")
	}

}
