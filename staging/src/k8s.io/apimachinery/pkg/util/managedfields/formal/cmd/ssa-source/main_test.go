/*
Copyright 2026 The Kubernetes Authors.

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

package main

import (
	"strings"
	"testing"
)

const signature = "func (s *Updater) addBackDanglingItems(merged, pruned *typed.TypedValue, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error)"

func sourceWith(body string) []byte {
	return []byte("package merge\n" + signature + " {\n" + body + "\n}\n")
}

func TestCompleteSyntaxAndErrorBranchesAreImported(t *testing.T) {
	program, err := importProgram(sourceWith(`converted, err := s.Converter.Convert(pruned, lastSet.APIVersion())
if err != nil {
  if s.Converter.IsMissingVersionError(err) { return merged, nil }
  return nil, fmt.Errorf("conversion: %v", err)
}
return converted, nil`))
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{"Convert", "IsMissingVersionError", "conversion: %v", "converted", "(.neNil"} {
		if !strings.Contains(program, required) {
			t.Fatalf("import dropped %q: %s", required, program)
		}
	}
}

func TestUnsupportedSyntaxCannotDisappear(t *testing.T) {
	for name, body := range map[string]string{
		"loop":         "for { return merged, nil }",
		"block shadow": "if merged != nil { merged := pruned; return merged, nil }; return merged, nil",
		"else":         "if merged != nil { return merged, nil } else { return pruned, nil }",
		"if init":      "if err := fail(); err != nil { return merged, err }; return pruned, nil",
		"call":         "return merge(merged, pruned), nil",
		"literal":      "return 42, nil",
		"named return": "return",
		"index":        "return objects[0], nil",
		"comparison":   "if merged == nil { return merged, nil }; return pruned, nil",
		"mutation":     "merged.field = pruned; return merged, nil",
		"global name":  "if globalError != nil { return pruned, nil }; return merged, nil",
		"unbound":      "return unknown, nil",
		"self binding": "unknown := unknown; return unknown, nil",
		"global write": "global = merged; return merged, nil",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := importProgram(sourceWith(body)); err == nil {
				t.Fatalf("unsupported syntax was accepted: %s", body)
			}
		})
	}
}

func TestSignatureChangesAreRejected(t *testing.T) {
	source := strings.Replace(string(sourceWith("return merged, nil")), "lastSet fieldpath.VersionedSet", "lastSet OtherType", 1)
	if _, err := importProgram([]byte(source)); err == nil {
		t.Fatal("changed parameter type was not rejected")
	}
}
