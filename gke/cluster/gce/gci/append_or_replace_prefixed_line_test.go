/*
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gci

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestAppendOrReplacePrefix(t *testing.T) {
	testCases := []struct {
		desc                string
		prefix              string
		suffix              string
		initialFileContents string
		want                string
	}{
		{
			desc:   "simple string and empty file",
			prefix: "hello",
			suffix: "world",
			want: `helloworld
`,
		},
		{
			desc:   "simple string and non empty file",
			prefix: "hello",
			suffix: "world",
			initialFileContents: `jelloworld
chelloworld
`,
			want: `jelloworld
chelloworld
helloworld
`,
		},
		{
			desc:   "simple string and file already contains prefix",
			prefix: "hello",
			suffix: "world",
			initialFileContents: `helloworld
helloworld
jelloworld
chelloworld
`,
			want: `jelloworld
chelloworld
helloworld
`,
		},
		{
			desc:   "simple string and file already contains prefix with content between the prefix and suffix",
			prefix: "hello",
			suffix: "world",
			initialFileContents: `hellocontentsworld
jelloworld
chelloworld
`,
			want: `jelloworld
chelloworld
helloworld
`,
		},
		{
			desc:   "simple string and file already contains prefix with prefix == suffix",
			prefix: "hello",
			suffix: "world",
			initialFileContents: `hellohello
jelloworld
chelloworld
`,
			want: `jelloworld
chelloworld
helloworld
`,
		},
		{
			desc:   "string with quotes and = and empty file",
			prefix: `'"$argon2id$v=19"'`,
			suffix: "admin",
			want: `"$argon2id$v=19"admin
`,
		},
		{
			desc:   "string with quotes and = and non empty file",
			prefix: `'"$argon2id$v=19"'`,
			suffix: "admin",
			initialFileContents: `jelloworld
chelloworld
`,
			want: `jelloworld
chelloworld
"$argon2id$v=19"admin
`,
		},
		{
			desc:   "string with quotes and = and file already contains prefix",
			prefix: `'"$argon2id$v=19"'`,
			suffix: "admin",
			initialFileContents: `"$argon2id$v=19"admin
"$argon2id$v=19"admin
helloworld
jelloworld
`,
			want: `helloworld
jelloworld
"$argon2id$v=19"admin
`,
		},
		{
			desc:   "string with quotes and = and file already contains prefix with content between the prefix and suffix",
			prefix: `'"$argon2id$v=19"'`,
			suffix: "admin",
			initialFileContents: `"$argon2id$v=19"contentsadmin
helloworld
jelloworld
`,
			want: `helloworld
jelloworld
"$argon2id$v=19"admin
`,
		},
		{
			desc:   "string with quotes and = and file already contains prefix with prefix == suffix",
			prefix: `'"$argon2id$v=19"'`,
			suffix: "admin",
			initialFileContents: `"$argon2id$v=19""$argon2id$v=19"
helloworld
jelloworld
`,
			want: `helloworld
jelloworld
"$argon2id$v=19"admin
`,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			f, err := ioutil.TempFile("", "append_or_replace_test")
			if err != nil {
				t.Fatalf("Failed to create temp file: %v", err)
			}
			defer os.Remove(f.Name())
			if _, err := f.WriteString(tc.initialFileContents); err != nil {
				t.Fatalf("Failed to write to file: %v", err)
			}
			f.Close()
			args := fmt.Sprintf("source configure-helper.sh; append_or_replace_prefixed_line %s %s %s", f.Name(), tc.prefix, tc.suffix)
			cmd := exec.Command("bash", "-c", args)
			stderr, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("Failed to run command: %v: %s", err, stderr)
			}
			got, err := ioutil.ReadFile(f.Name())
			if err != nil {
				t.Fatalf("Failed to read file contents: %v", err)
			}
			if diff := cmp.Diff(string(got), tc.want); diff != "" {
				t.Errorf("File contents: got=%s, want=%s, diff=%s", got, tc.want, diff)
			}
		})
	}

}
