/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	openAPIGenTag = "// +k8s:openapi-gen"

	staging = "staging/src/"
)

// walkGenerated updates the rule for kubernetes' OpenAPI generated file.
// This involves reading all go files in the source tree and looking for the
// "+k8s:openapi-gen" tag. If present, then that package must be supplied to
// the genrule.
func (v *Vendorer) walkGenerated() error {
	if !v.cfg.K8sOpenAPIGen {
		return nil
	}
	v.managedAttrs = append(v.managedAttrs, "openapi_targets", "vendor_targets")
	paths, err := v.findOpenAPI(".")
	if err != nil {
		return err
	}
	return v.addGeneratedOpenAPIRule(paths)
}

// findOpenAPI searches for all packages under root that request OpenAPI. It
// returns the go import paths. It does not follow symlinks.
func (v *Vendorer) findOpenAPI(root string) ([]string, error) {
	for _, r := range v.skippedPaths {
		if r.MatchString(root) {
			return nil, nil
		}
	}
	finfos, err := ioutil.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var res []string
	var includeMe bool
	for _, finfo := range finfos {
		path := filepath.Join(root, finfo.Name())
		if finfo.IsDir() && (finfo.Mode()&os.ModeSymlink == 0) {
			children, err := v.findOpenAPI(path)
			if err != nil {
				return nil, err
			}
			res = append(res, children...)
		} else if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			b, err := ioutil.ReadFile(path)
			if err != nil {
				return nil, err
			}
			if bytes.Contains(b, []byte(openAPIGenTag)) {
				includeMe = true
			}
		}
	}
	if includeMe {
		pkg, err := v.ctx.ImportDir(filepath.Join(v.root, root), 0)
		if err != nil {
			return nil, err
		}
		res = append(res, pkg.ImportPath)
	}
	return res, nil
}

// addGeneratedOpenAPIRule updates the pkg/generated/openapi go_default_library
// rule with the automanaged openapi_targets and vendor_targets.
func (v *Vendorer) addGeneratedOpenAPIRule(paths []string) error {
	var openAPITargets []string
	var vendorTargets []string
	baseImport := v.cfg.GoPrefix + "/"
	for _, p := range paths {
		if !strings.HasPrefix(p, baseImport) {
			return fmt.Errorf("openapi-gen path outside of %s: %s", v.cfg.GoPrefix, p)
		}
		np := p[len(baseImport):]
		if strings.HasPrefix(np, staging) {
			vendorTargets = append(vendorTargets, np[len(staging):])
		} else {
			openAPITargets = append(openAPITargets, np)
		}
	}
	sort.Strings(openAPITargets)
	sort.Strings(vendorTargets)

	pkgPath := filepath.Join("pkg", "generated", "openapi")
	// If we haven't walked this package yet, walk it so there is a go_library rule to modify
	if len(v.newRules[pkgPath]) == 0 {
		if err := v.updateSinglePkg(pkgPath); err != nil {
			return err
		}
	}
	for _, r := range v.newRules[pkgPath] {
		if r.Name() == "go_default_library" {
			r.SetAttr("openapi_targets", asExpr(openAPITargets))
			r.SetAttr("vendor_targets", asExpr(vendorTargets))
			break
		}
	}
	return nil
}
