/*
Copyright The Kubernetes Authors.

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

package apirequirements

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

func TestFeatureGateAPIRequirementsWellFormed(t *testing.T) {
	registeredFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	tests := []struct {
		name         string
		requirements serverstorage.FeatureGateAPIRequirements
	}{
		{
			name:         "generic control plane requirements",
			requirements: DefaultForGenericControlPlane(),
		},
		{
			name:         "kube-apiserver requirements",
			requirements: DefaultForKubeAPIServer(),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for feature, resources := range tc.requirements {
				if _, ok := registeredFeatures[feature]; !ok {
					t.Errorf("feature gate %q declares API requirements but is not registered in the DefaultFeatureGate", feature)
				}
				if len(resources) == 0 {
					t.Errorf("feature gate %q declares an empty requirement list; omit the entry instead", feature)
				}
				seen := sets.New[schema.GroupResource]()
				for _, gr := range resources {
					if gr.Resource == "" {
						t.Errorf("feature gate %q has a malformed requirement %#v: resource must be set", feature, gr)
					}
					if seen.Has(gr) {
						t.Errorf("feature gate %q lists duplicate requirement %s", feature, gr)
					}
					seen.Insert(gr)
				}
			}
		})
	}

	// kube-apiserver serves a superset of the generic control plane, so a gate validated on a
	// generic control plane must be validated identically on kube-apiserver.
	kube := DefaultForKubeAPIServer()
	for feature, resources := range DefaultForGenericControlPlane() {
		if !reflect.DeepEqual(kube[feature], resources) {
			t.Errorf("feature gate %q requires %v generically but %v in kube-apiserver; the kube requirements must include the generic ones", feature, resources, kube[feature])
		}
	}
}

func registrySourceDir(t *testing.T) string {
	t.Helper()
	return filepath.Join(repoRoot(t), "pkg", "registry")
}

func repoRoot(t *testing.T) string {
	t.Helper()

	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getting working directory: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("found no go.mod above the working directory, so the repository root is unknown")
		}
		dir = parent
	}
}

// featuresRequiringAPIsTheyDoNotGate lists feature gates that require an API resource whose
// installation is not itself guarded by that gate, so TestFeatureGateAPIRequirementsMatchProviders
// cannot find the requirement in the provider source. Add a gate here only when its feature
// consumes an API that some other component installs unconditionally.
var featuresRequiringAPIsTheyDoNotGate = sets.New[featuregate.Feature]()

func TestFeatureGateAPIRequirementsMatchProviders(t *testing.T) {
	registeredFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	gatedInProviders := map[featuregate.Feature]sets.Set[string]{}
	for _, gated := range gatedResourcesInProviders(t) {
		if _, ok := registeredFeatures[gated.feature]; !ok {
			t.Errorf("provider source gates %q on %q, which is not a registered feature gate; the constant name probably no longer matches the gate it holds", gated.resource, gated.feature)
			continue
		}
		if gatedInProviders[gated.feature] == nil {
			gatedInProviders[gated.feature] = sets.New[string]()
		}
		gatedInProviders[gated.feature].Insert(gated.resource)
	}
	if len(gatedInProviders) == 0 {
		t.Fatalf("found no feature-gated resources in the pkg/registry providers; the provider idiom this test matches has changed and the test is no longer checking anything")
	}

	declared := map[featuregate.Feature]sets.Set[string]{}
	for feature, resources := range DefaultForKubeAPIServer() {
		declared[feature] = sets.New[string]()
		for _, gr := range resources {
			declared[feature].Insert(gr.Resource)
		}
	}

	allFeatures := sets.New[featuregate.Feature]()
	allFeatures.Insert(slices.Collect(maps.Keys(gatedInProviders))...)
	allFeatures.Insert(slices.Collect(maps.Keys(declared))...)

	for _, feature := range slices.Sorted(maps.Keys(allFeatures)) {
		t.Run(string(feature), func(t *testing.T) {
			if undeclared := gatedInProviders[feature].Difference(declared[feature]); undeclared.Len() > 0 {
				t.Errorf("a provider installs %v only when %q is enabled, but %q does not require them; add them to DefaultForKubeAPIServer so enabling the gate without the API fails at startup", sets.List(undeclared), feature, feature)
			}
			if featuresRequiringAPIsTheyDoNotGate.Has(feature) {
				return
			}
			if unguarded := declared[feature].Difference(gatedInProviders[feature]); unguarded.Len() > 0 {
				t.Errorf("%q requires %v, but no provider gates those resources on it; drop the stale requirement, or record the gate in featuresRequiringAPIsTheyDoNotGate if the API is installed elsewhere", feature, sets.List(unguarded))
			}
		})
	}
}

type gatedResource struct {
	feature  featuregate.Feature
	resource string
}

// gatedResourcesInProviders extracts every feature-gated resource from the REST storage providers
// by matching the idiom they all follow:
//
//	if resource := "clustertrustbundles"; apiResourceConfigSource.ResourceEnabled(...) {
//		if utilfeature.DefaultFeatureGate.Enabled(features.ClusterTrustBundle) {
//			storage[resource] = bundleStorage
//
// Gates that only add a subresource, such as pods/resize, are not requirements: the parent resource
// is served either way, and requirements cannot address a subresource.
func gatedResourcesInProviders(t *testing.T) []gatedResource {
	t.Helper()

	dir := registrySourceDir(t)

	var found []gatedResource
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() || filepath.Base(filepath.Dir(path)) != "rest" {
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return fmt.Errorf("parsing %s: %w", path, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			resource, ok := resourceGuard(n)
			if !ok {
				return true
			}
			for _, feature := range gatesEnabling(n.(*ast.IfStmt).Body) {
				found = append(found, gatedResource{feature: feature, resource: resource})
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("walking %s: %v", dir, err)
	}
	return found
}

func resourceGuard(n ast.Node) (string, bool) {
	ifStmt, ok := n.(*ast.IfStmt)
	if !ok || ifStmt.Init == nil {
		return "", false
	}
	assign, ok := ifStmt.Init.(*ast.AssignStmt)
	if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
		return "", false
	}
	if lhs, ok := assign.Lhs[0].(*ast.Ident); !ok || lhs.Name != "resource" {
		return "", false
	}
	lit, ok := assign.Rhs[0].(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING || !callsResourceEnabled(ifStmt.Cond) {
		return "", false
	}
	resource, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return resource, true
}

func callsResourceEnabled(cond ast.Expr) bool {
	found := false
	ast.Inspect(cond, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "ResourceEnabled" {
			found = true
			return false
		}
		return true
	})
	return found
}

func gatesEnabling(body *ast.BlockStmt) []featuregate.Feature {
	var features []featuregate.Feature
	ast.Inspect(body, func(n ast.Node) bool {
		ifStmt, ok := n.(*ast.IfStmt)
		if !ok {
			return true
		}
		if _, nested := resourceGuard(ifStmt); nested {
			return false
		}
		if !installsBareResource(ifStmt.Body) {
			return true
		}
		features = append(features, gatesRequiredBy(ifStmt.Cond)...)
		return true
	})
	return features
}

func installsBareResource(body *ast.BlockStmt) bool {
	installs := false
	ast.Inspect(body, func(n ast.Node) bool {
		if ifStmt, ok := n.(*ast.IfStmt); ok {
			if _, nested := resourceGuard(ifStmt); nested {
				return false
			}
			return true
		}
		assign, ok := n.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 {
			return true
		}
		index, ok := assign.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return true
		}
		if ident, ok := index.Index.(*ast.Ident); ok && ident.Name == "resource" {
			installs = true
			return false
		}
		return true
	})
	return installs
}

func gatesRequiredBy(cond ast.Expr) []featuregate.Feature {
	switch expr := cond.(type) {
	case *ast.BinaryExpr:
		if expr.Op != token.LAND {
			return nil
		}
		return append(gatesRequiredBy(expr.X), gatesRequiredBy(expr.Y)...)
	case *ast.CallExpr:
		sel, ok := expr.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Enabled" || len(expr.Args) != 1 {
			return nil
		}
		// Only kube gates, imported as features, can appear in the declared requirements.
		gate, ok := expr.Args[0].(*ast.SelectorExpr)
		if !ok {
			return nil
		}
		if pkg, ok := gate.X.(*ast.Ident); !ok || pkg.Name != "features" {
			return nil
		}
		return []featuregate.Feature{featuregate.Feature(gate.Sel.Name)}
	}
	return nil
}
