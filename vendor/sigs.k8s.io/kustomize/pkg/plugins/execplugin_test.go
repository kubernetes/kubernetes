/*
Copyright 2019 The Kubernetes Authors.

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

package plugins

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
)

func TestExecPluginConfig(t *testing.T) {
	path := "/app"
	rf := resmap.NewFactory(
		resource.NewFactory(
			kunstruct.NewKunstructuredFactoryImpl()))
	ldr := loadertest.NewFakeLoader(path)
	pluginConfig := rf.RF().FromMap(
		map[string]interface{}{
			"apiVersion": "someteam.example.com/v1",
			"kind":       "SedTransformer",
			"metadata": map[string]interface{}{
				"name": "some-random-name",
			},
			"argsOneLiner": "one two",
			"argsFromFile": "sed-input.txt",
		})

	ldr.AddFile("/app/sed-input.txt", []byte(`
s/$FOO/foo/g
s/$BAR/bar/g
 \ \ \ 
`))

	p := NewExecPlugin(
		AbsolutePluginPath(
			DefaultPluginConfig(),
			pluginConfig.OrgId()))

	yaml, err := pluginConfig.AsYAML()
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	p.Config(ldr, rf, yaml)

	expected := "/kustomize/plugin/someteam.example.com/v1/sedtransformer/SedTransformer"
	if !strings.HasSuffix(p.path, expected) {
		t.Fatalf("expected suffix '%s', got '%s'", expected, p.path)
	}

	expected = `apiVersion: someteam.example.com/v1
argsFromFile: sed-input.txt
argsOneLiner: one two
kind: SedTransformer
metadata:
  name: some-random-name
`
	if expected != string(p.cfg) {
		t.Fatalf("expected cfg '%s', got '%s'", expected, string(p.cfg))

	}
	if len(p.args) != 5 {
		t.Fatalf("unexpected arg len %d, %v", len(p.args), p.args)
	}
	if p.args[0] != "one" ||
		p.args[1] != "two" ||
		p.args[2] != "s/$FOO/foo/g" ||
		p.args[3] != "s/$BAR/bar/g" ||
		p.args[4] != "\\ \\ \\ " {
		t.Fatalf("unexpected arg array: %v", p.args)
	}
}
