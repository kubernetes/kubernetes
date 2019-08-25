// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kusttest_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/k8sdeps/transformer"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/plugins"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/target"
	"sigs.k8s.io/kustomize/pkg/transformers/config/defaultconfig"
	"sigs.k8s.io/kustomize/pkg/types"
)

// KustTestHarness helps test kustomization generation and transformation.
type KustTestHarness struct {
	t   *testing.T
	rf  *resmap.Factory
	ldr loadertest.FakeLoader
	pl  *plugins.Loader
}

func NewKustTestHarness(t *testing.T, path string) *KustTestHarness {
	return NewKustTestHarnessWithPluginConfig(
		t, path, plugins.DefaultPluginConfig())
}

func NewKustTestPluginHarness(t *testing.T, path string) *KustTestHarness {
	return NewKustTestHarnessWithPluginConfig(
		t, path, plugins.ActivePluginConfig())
}

func NewKustTestHarnessWithPluginConfig(
	t *testing.T, path string,
	pc *types.PluginConfig) *KustTestHarness {
	return NewKustTestHarnessFull(t, path, loader.RestrictionRootOnly, pc)
}

func NewKustTestHarnessFull(
	t *testing.T, path string,
	lr loader.LoadRestrictorFunc, pc *types.PluginConfig) *KustTestHarness {
	rf := resmap.NewFactory(resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl()))
	return &KustTestHarness{
		t:   t,
		rf:  rf,
		ldr: loadertest.NewFakeLoaderWithRestrictor(lr, path),
		pl:  plugins.NewLoader(pc, rf)}
}

func (th *KustTestHarness) MakeKustTarget() *target.KustTarget {
	kt, err := target.NewKustTarget(
		th.ldr, th.rf, transformer.NewFactoryImpl(), th.pl)
	if err != nil {
		th.t.Fatalf("Unexpected construction error %v", err)
	}
	return kt
}

func (th *KustTestHarness) WriteF(dir string, content string) {
	err := th.ldr.AddFile(dir, []byte(content))
	if err != nil {
		th.t.Fatalf("failed write to %s; %v", dir, err)
	}
}

func (th *KustTestHarness) WriteK(dir string, content string) {
	th.WriteF(filepath.Join(dir, pgmconfig.KustomizationFileNames[0]), `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
`+content)
}

func (th *KustTestHarness) RF() *resource.Factory {
	return th.rf.RF()
}

func (th *KustTestHarness) FromMap(m map[string]interface{}) *resource.Resource {
	return th.rf.RF().FromMap(m)
}

func (th *KustTestHarness) FromMapAndOption(m map[string]interface{}, args *types.GeneratorArgs, option *types.GeneratorOptions) *resource.Resource {
	return th.rf.RF().FromMapAndOption(m, args, option)
}

func (th *KustTestHarness) WriteDefaultConfigs(fName string) {
	m := defaultconfig.GetDefaultFieldSpecsAsMap()
	var content []byte
	for _, tCfg := range m {
		content = append(content, []byte(tCfg)...)
	}
	err := th.ldr.AddFile(fName, content)
	if err != nil {
		th.t.Fatalf("unable to add file %s", fName)
	}
}

func (th *KustTestHarness) LoadAndRunGenerator(
	config string) resmap.ResMap {
	res, err := th.rf.RF().FromBytes([]byte(config))
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	g, err := th.pl.LoadGenerator(th.ldr, res)
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	rm, err := g.Generate()
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	return rm
}

func (th *KustTestHarness) LoadAndRunTransformer(
	config, input string) resmap.ResMap {
	resMap, err := th.runTransformer(config, input)
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	return resMap
}

func (th *KustTestHarness) ErrorFromLoadAndRunTransformer(
	config, input string) error {
	_, err := th.runTransformer(config, input)
	return err
}

func (th *KustTestHarness) runTransformer(
	config, input string) (resmap.ResMap, error) {
	transConfig, err := th.rf.RF().FromBytes([]byte(config))
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	resMap, err := th.rf.NewResMapFromBytes([]byte(input))
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	g, err := th.pl.LoadTransformer(th.ldr, transConfig)
	if err != nil {
		th.t.Fatalf("Err: %v", err)
	}
	err = g.Transform(resMap)
	return resMap, err
}

func tabToSpace(input string) string {
	var result []string
	for _, i := range input {
		if i == 9 {
			result = append(result, "  ")
		} else {
			result = append(result, string(i))
		}
	}
	return strings.Join(result, "")
}

func convertToArray(x string) ([]string, int) {
	a := strings.Split(strings.TrimSuffix(x, "\n"), "\n")
	maxLen := 0
	for i, v := range a {
		z := tabToSpace(v)
		if len(z) > maxLen {
			maxLen = len(z)
		}
		a[i] = z
	}
	return a, maxLen
}

func hint(a, b string) string {
	if a == b {
		return " "
	}
	return "X"
}

func (th *KustTestHarness) AssertActualEqualsExpected(
	m resmap.ResMap, expected string) {
	if m == nil {
		th.t.Fatalf("Map should not be nil.")
	}
	// Ignore leading linefeed in expected value
	// to ease readability of tests.
	if len(expected) > 0 && expected[0] == 10 {
		expected = expected[1:]
	}
	actual, err := m.AsYaml()
	if err != nil {
		th.t.Fatalf("Unexpected err: %v", err)
	}
	if string(actual) != expected {
		th.reportDiffAndFail(actual, expected)
	}
}

// Pretty printing of file differences.
func (th *KustTestHarness) reportDiffAndFail(actual []byte, expected string) {
	sE, maxLen := convertToArray(expected)
	sA, _ := convertToArray(string(actual))
	fmt.Println("===== ACTUAL BEGIN ========================================")
	fmt.Print(string(actual))
	fmt.Println("===== ACTUAL END ==========================================")
	format := fmt.Sprintf("%%s  %%-%ds %%s\n", maxLen+4)
	limit := 0
	if len(sE) < len(sA) {
		limit = len(sE)
	} else {
		limit = len(sA)
	}
	fmt.Printf(format, " ", "EXPECTED", "ACTUAL")
	fmt.Printf(format, " ", "--------", "------")
	for i := 0; i < limit; i++ {
		fmt.Printf(format, hint(sE[i], sA[i]), sE[i], sA[i])
	}
	if len(sE) < len(sA) {
		for i := len(sE); i < len(sA); i++ {
			fmt.Printf(format, "X", "", sA[i])
		}
	} else {
		for i := len(sA); i < len(sE); i++ {
			fmt.Printf(format, "X", sE[i], "")
		}
	}
	th.t.Fatalf("Expected not equal to actual")
}
