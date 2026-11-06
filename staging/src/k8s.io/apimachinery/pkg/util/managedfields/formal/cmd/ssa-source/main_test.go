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
		"else if":      "if merged != nil { return merged, nil } else if pruned != nil { return pruned, nil }",
		"if init":      "if err := fail(); err != nil { return merged, err }; return pruned, nil",
		"call":         "return merge(merged, pruned), nil",
		"literal":      "return 42, nil",
		"named return": "return",
		"index":        "return objects[0], nil",
		"comparison":   "if merged == pruned { return merged, nil }; return pruned, nil",
		"mutation":     "merged.field = pruned; return merged, nil",
		"global name":  "if globalError != nil { return pruned, nil }; return merged, nil",
		"unbound":      "return unknown, nil",
		"self binding": "unknown := unknown; return unknown, nil",
		"global write": "global = merged; return merged, nil",
		"zero type":    "var count int; return merged, nil",
		"zero init":    "var err error = fail(); return merged, err",
		"zero shadow":  "var merged error; return pruned, merged",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := importProgram(sourceWith(body)); err == nil {
				t.Fatalf("unsupported syntax was accepted: %s", body)
			}
		})
	}
}

func TestApplyImportsFilterBranchAndActorMapWrite(t *testing.T) {
	source := []byte("package merge\n" + functions[4].signature + ` {
var ignoreFilter fieldpath.Filter
if s.IgnoredFields != nil && s.IgnoreFilter != nil { return nil, nil, fmt.Errorf("both") }
if s.IgnoredFields != nil {
  ignoreFilter = fieldpath.NewExcludeSetFilter(s.IgnoredFields[version])
} else {
  ignoreFilter = s.IgnoreFilter[version]
}
set, err := configObject.ToFieldSet()
if err != nil { return nil, fieldpath.ManagedFields{}, err }
if ignoreFilter != nil { set = ignoreFilter.Filter(set) }
managers[manager] = fieldpath.NewVersionedSet(set, version, true)
return liveObject, managers, nil
}`)
	program, err := importFunction(source, functions[4])
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{`(.declareNil "ignoreFilter")`, "(.branchElse", "(.andAlso", `(.mapSet "managers"`, "NewExcludeSetFilter", "(.flag true)", "(.emptyManagedMap)"} {
		if !strings.Contains(program, expected) {
			t.Fatalf("Apply import dropped %q: %s", expected, program)
		}
	}
}

func TestMutableApplyMapStillRejectsAliasesAndIterationWrites(t *testing.T) {
	for _, body := range []string{
		"alias := managers; return liveObject, managers, nil",
		"for key, entry := range managers { managers[key] = entry }; return liveObject, managers, nil",
	} {
		source := []byte("package merge\n" + functions[4].signature + " {\n" + body + "\n}")
		if _, err := importFunction(source, functions[4]); err == nil {
			t.Fatalf("unmodeled mutable-map behavior was accepted: %s", body)
		}
	}
}

func TestBooleanShadowKeepsItsLexicalBinding(t *testing.T) {
	source := []byte("package merge\n" + functions[4].signature + ` {
true := force
set, err := configObject.ToFieldSet()
managers[manager] = fieldpath.NewVersionedSet(set, version, true)
return liveObject, managers, err
}`)
	program, err := importFunction(source, functions[4])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(program, `(.var "true$1")`) || strings.Contains(program, "(.flag true)") {
		t.Fatalf("shadowed true was replaced by a Boolean literal: %s", program)
	}
}

func TestLexicalShadowDoesNotChangeOuterBinding(t *testing.T) {
	program, err := importProgram(sourceWith("if merged != nil { merged := pruned; return merged, nil }; return merged, nil"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(program, `"merged$1"`) || !strings.HasSuffix(program, "(.ret [(.var \"merged\"), (.var \"nil\")])\n]") {
		t.Fatalf("lexical shadow leaked into the outer return: %s", program)
	}
}

func TestOwnedVersionImportsTypedZeroAndVersionedErrors(t *testing.T) {
	source := []byte("package merge\n" + functions[1].signature + ` {
var err error
merged, err = s.Converter.Convert(merged, version)
if err != nil { return nil, nil, fmt.Errorf("conversion at %v: %v", version, err) }
return merged, pruned, nil
}`)
	program, err := importFunction(source, functions[1])
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{`(.declareError "err")`, "conversion at %v: %v", `(.var "version")`} {
		if !strings.Contains(program, required) {
			t.Fatalf("import dropped %q: %s", required, program)
		}
	}
}

func TestSignatureChangesAreRejected(t *testing.T) {
	source := strings.Replace(string(sourceWith("return merged, nil")), "lastSet fieldpath.VersionedSet", "lastSet OtherType", 1)
	if _, err := importProgram([]byte(source)); err == nil {
		t.Fatal("changed parameter type was not rejected")
	}
}

func TestMapsCannotAcquireUnmodeledAliasesOrIterationMutations(t *testing.T) {
	for name, body := range map[string]string{
		"input alias":  "alias := managedFields; return pruned, nil",
		"local alias":  "fields := map[fieldpath.APIVersion]*fieldpath.Set{}; alias := fields; return pruned, nil",
		"input write":  `managedFields["v1"] = fieldpath.NewSet(); return pruned, nil`,
		"range delete": "fields := map[fieldpath.APIVersion]*fieldpath.Set{}; for version, managed := range fields { delete(fields, version) }; return pruned, nil",
		"range write":  "fields := map[fieldpath.APIVersion]*fieldpath.Set{}; for version, managed := range fields { fields[version] = managed }; return pruned, nil",
		"nil shadow":   "nil := merged; if pruned == nil { return merged, nil }; return pruned, nil",
	} {
		t.Run(name, func(t *testing.T) {
			source := []byte("package merge\n" + functions[3].signature + " {\n" + body + "\n}")
			if _, err := importFunction(source, functions[3]); err == nil {
				t.Fatalf("unsupported alias or map behavior was accepted: %s", body)
			}
		})
	}
}

func TestImportBindingsCannotBeReplacedUnderTheSameName(t *testing.T) {
	source := `package merge
import (
 "fmt"
 "sigs.k8s.io/structured-merge-diff/v6/typed"
 fieldpath "example.com/unrelated"
)
`
	if err := validateImports([]byte(source)); err == nil {
		t.Fatal("unrelated package was accepted as the fieldpath primitive implementation")
	}
}
