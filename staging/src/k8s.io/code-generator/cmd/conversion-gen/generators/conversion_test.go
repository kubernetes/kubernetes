/*
Copyright 2026 The Kubernetes Authors.
*/

package generators

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/code-generator/cmd/conversion-gen/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
)

func TestValidateGroupHubs(t *testing.T) {
	cases := []struct {
		name          string
		pkgs          []*types.Package
		inputs        []string
		pkgToPeers    map[string][]string
		pkgToExternal map[string]string
		requireHub    bool
		wantErr       string
		wantReport    string // substring the violation report must contain. If empty, an empty report is expected.
	}{
		{
			name: "success: 1 hub, memory identical",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType"},
				}, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v2", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.String}}, // Different, but not hub
				}),
			},
			inputs: []string{"example.com/pkg/v1", "example.com/pkg/v2"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
				"example.com/pkg/v2": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
				"example.com/pkg/v2": "example.com/pkg/v2",
			},
			requireHub: false,
			wantReport: "",
		},
		{
			name: "success: groupName is used properly",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v1", []string{"+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType"},
				}, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v2", []string{"+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.String}},
				}),
			},
			inputs: []string{"example.com/pkg/v1", "example.com/pkg/v2"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
				"example.com/pkg/v2": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
				"example.com/pkg/v2": "example.com/pkg/v2",
			},
			requireHub: true,
			wantReport: "",
		},
		{
			name: "failure: multiple hubs",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType"},
				}, nil),
				makePkg("example.com/pkg/v2", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType"},
				}, nil),
			},
			inputs: []string{"example.com/pkg/v1", "example.com/pkg/v2"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
				"example.com/pkg/v2": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
				"example.com/pkg/v2": "example.com/pkg/v2",
			},
			requireHub: false,
			wantErr:    `multiple hub types`,
		},
		{
			name: "failure: hub not memory identical",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType"},
				}, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.String}}, // Different type
				}),
			},
			inputs: []string{"example.com/pkg/v1"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
			},
			requireHub: false,
			wantReport: `hub_memory_identity,example.com/pkg/v1,TypeA,X`,
		},
		{
			name: "success: internal type opt-out",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, map[string][]string{
					"TypeA": {"+k8s:hubType=false"}, // Opt-out on internal type!
				}, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.Int32}},
				}),
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, map[string][]types.Member{
					"TypeA": {{Name: "X", Type: types.String}}, // Different, and not hub
				}),
			},
			inputs: []string{"example.com/pkg/v1"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
			},
			requireHub: true,
			wantReport: "",
		},
		{
			name: "failure: 0 hubs, with --require-hub-types",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, nil),
				makePkg("example.com/pkg/v2", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, nil),
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, nil, nil),
			},
			inputs: []string{"example.com/pkg/v1", "example.com/pkg/v2"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
				"example.com/pkg/v2": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
				"example.com/pkg/v2": "example.com/pkg/v2",
			},
			requireHub: true,
			wantReport: `hub_type_missing,example.com/pkg/internal,TypeA,`,
		},
		{
			name: "success: 0 hubs, no --require-hub-types",
			pkgs: []*types.Package{
				makePkg("example.com/pkg/v1", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, nil),
				makePkg("example.com/pkg/v2", []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, []string{"TypeA"}, nil, nil),
				makePkg("example.com/pkg/internal", []string{"+groupName=testgroup.k8s.io"}, []string{"TypeA"}, nil, nil),
			},
			inputs: []string{"example.com/pkg/v1", "example.com/pkg/v2"},
			pkgToPeers: map[string][]string{
				"example.com/pkg/v1": {"example.com/pkg/internal"},
				"example.com/pkg/v2": {"example.com/pkg/internal"},
			},
			pkgToExternal: map[string]string{
				"example.com/pkg/v1": "example.com/pkg/v1",
				"example.com/pkg/v2": "example.com/pkg/v2",
			},
			requireHub: false,
			wantReport: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			universe := types.Universe{}
			for _, pkg := range tc.pkgs {
				universe[pkg.Path] = pkg
			}
			context := &generator.Context{Universe: universe}
			report := filepath.Join(t.TempDir(), "report")
			generatorArgs := argsWithReport(report)
			if tc.requireHub {
				generatorArgs.LintRules = []string{requireHubTypesLintRule}
			}

			err := validateAndCheckRules(context, generatorArgs, tc.inputs, tc.pkgToPeers, tc.pkgToExternal, equalMemoryTypes{})
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			got := readReport(t, report)
			if tc.wantReport == "" {
				if got != "" {
					t.Fatalf("expected empty report, got:\n%s", got)
				}
				return
			}
			if !strings.Contains(got, tc.wantReport) {
				t.Fatalf("report missing %q, got:\n%s", tc.wantReport, got)
			}
		})
	}
}

func TestValidateGroupHubsManualConversion(t *testing.T) {
	nestedInternal := &types.Type{Name: types.Name{Package: "example.com/pkg/internal", Name: "Nested"}, Kind: types.Struct, Members: []types.Member{{Name: "X", Type: types.Int32}}}
	nestedV1 := &types.Type{Name: types.Name{Package: "example.com/pkg/v1", Name: "Nested"}, Kind: types.Struct, Members: []types.Member{{Name: "X", Type: types.Int32}}}
	ptrInternal := &types.Type{Name: types.Name{Name: "*Nested"}, Kind: types.Pointer, Elem: nestedInternal}
	ptrV1 := &types.Type{Name: types.Name{Name: "*Nested"}, Kind: types.Pointer, Elem: nestedV1}

	internalPkg := &types.Package{Path: "example.com/pkg/internal", Name: "internal", Comments: []string{"+groupName=testgroup.k8s.io"}, Types: map[string]*types.Type{
		"TypeA":  {Name: types.Name{Package: "example.com/pkg/internal", Name: "TypeA"}, Kind: types.Struct, Members: []types.Member{{Name: "N", Type: ptrInternal}}},
		"Nested": nestedInternal,
	}}
	v1Pkg := &types.Package{Path: "example.com/pkg/v1", Name: "v1", Comments: []string{"+groupName=testgroup.k8s.io", "+k8s:conversion-gen=example.com/pkg/internal"}, Types: map[string]*types.Type{
		"TypeA":  {Name: types.Name{Package: "example.com/pkg/v1", Name: "TypeA"}, Kind: types.Struct, CommentLines: []string{"+k8s:hubType"}, Members: []types.Member{{Name: "N", Type: ptrV1}}},
		"Nested": nestedV1,
	}}
	context := &generator.Context{Universe: types.Universe{internalPkg.Path: internalPkg, v1Pkg.Path: v1Pkg}}
	inputs := []string{"example.com/pkg/v1"}
	pkgToPeers := map[string][]string{"example.com/pkg/v1": {"example.com/pkg/internal"}}
	pkgToExternal := map[string]string{"example.com/pkg/v1": "example.com/pkg/v1"}

	// memory-identical with no manual conversion:
	report := filepath.Join(t.TempDir(), "report")
	if err := validateAndCheckRules(context, argsWithReport(report), inputs, pkgToPeers, pkgToExternal, equalMemoryTypes{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := readReport(t, report); got != "" {
		t.Fatalf("expected no violation for identical hub, got:\n%s", got)
	}
	// manual conversion:
	eq := equalMemoryTypes{}
	eq.Skip(nestedV1, nestedInternal)
	report2 := filepath.Join(t.TempDir(), "report")
	if err := validateAndCheckRules(context, argsWithReport(report2), inputs, pkgToPeers, pkgToExternal, eq); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := readReport(t, report2); !strings.Contains(got, "hub_memory_identity,example.com/pkg/v1,TypeA,N") {
		t.Fatalf("expected manual-conversion Skip to be reported at TypeA.N, got:\n%s", got)
	}
}

func makePkg(path string, comments []string, typeNames []string, typeComments map[string][]string, members map[string][]types.Member) *types.Package {
	pkg := &types.Package{
		Path:     path,
		Name:     path,
		Comments: comments,
		Types:    map[string]*types.Type{},
	}
	for _, tname := range typeNames {
		var m []types.Member
		if members != nil {
			m = members[tname]
		}
		var tc []string
		if typeComments != nil {
			tc = typeComments[tname]
		}
		pkg.Types[tname] = &types.Type{
			Name:         types.Name{Package: path, Name: tname},
			Kind:         types.Struct,
			CommentLines: tc,
			Members:      m,
		}
	}
	return pkg
}

func readReport(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading report: %v", err)
	}
	return string(b)
}

func argsWithReport(path string) *args.Args {
	a := &args.Args{}
	a.ReportFilename = path
	return a
}
