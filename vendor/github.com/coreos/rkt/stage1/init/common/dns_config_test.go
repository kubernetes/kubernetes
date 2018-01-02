// Copyright 2017 The rkt Authors
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

package common

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestAddHostsEntry(t *testing.T) {
	for i, tt := range []struct {
		contents string
		ip       string
		hostname string
		expected string
	}{
		{ // Blank file
			"",
			"127.0.0.1",
			"entry",
			"127.0.0.1\tentry\n",
		},
		{ // One line, found
			"127.0.0.1 localhost localhost.localdomain\n",
			"127.0.0.1",
			"testing",
			"127.0.0.1 localhost localhost.localdomain testing\n",
		},
		{ // Some lines, not found
			"4.2.2.2 hello goodbye\n\n\n",
			"127.0.0.1",
			"testing",
			"4.2.2.2 hello goodbye\n\n\n127.0.0.1\ttesting\n",
		},
		{ // IP address duplicated
			"127.0.0.1 hello\n127.0.0.1 broken\n",
			"127.0.0.1",
			"testing",
			"127.0.0.1 hello testing\n127.0.0.1 broken\n",
		},
		{ // Lots of lines, found.
			"4.2.2.2 hello goodbye\n\n\n127.0.0.1 foo\n\n#words\n",
			"127.0.0.1",
			"testing",
			"4.2.2.2 hello goodbye\n\n\n127.0.0.1 foo testing\n\n#words\n",
		},
	} {
		tmpfile, err := ioutil.TempFile("", "common-test-hosts")
		if err != nil {
			t.Fatal("failed to write temporary file", err)
		}

		defer os.Remove(tmpfile.Name())

		if _, err := tmpfile.Write([]byte(tt.contents)); err != nil {
			t.Fatal("failed to write temporary file", err)
		}

		if err := tmpfile.Close(); err != nil {
			t.Fatal("failed to close temporary file", err)
		}

		if err != AddHostsEntry(tmpfile.Name(), tt.ip, tt.hostname) {
			t.Fatal("failed to write new hostname", err)
		}

		newContents, err := ioutil.ReadFile(tmpfile.Name())

		if tt.expected != string(newContents) {
			t.Fatalf("Test case %d failed: expected contents %v, actual %v", i, tt.expected, string(newContents))
		}
	}
}
