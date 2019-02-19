/*
Copyright 2018 The Kubernetes Authors.

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

package target

import (
	"fmt"
	"log"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/pkg/types"
)

// ResAccumulator accumulates resources and the rules
// used to customize those resources.
// TODO(monopole): Move to "accumulator" package and make members private.
// This will make a better separation between KustTarget, which should
// be mainly concerned with data loading, and this class, which could
// become the home of all transformation data and logic.
type ResAccumulator struct {
	resMap  resmap.ResMap
	tConfig *config.TransformerConfig
	varSet  types.VarSet
}

func MakeEmptyAccumulator() *ResAccumulator {
	ra := &ResAccumulator{}
	ra.resMap = make(resmap.ResMap)
	ra.tConfig = &config.TransformerConfig{}
	ra.varSet = types.VarSet{}
	return ra
}

// ResMap returns a copy of the internal resMap.
func (ra *ResAccumulator) ResMap() resmap.ResMap {
	result := make(resmap.ResMap)
	for k, v := range ra.resMap {
		result[k] = v
	}
	return result
}

func (ra *ResAccumulator) MergeResourcesWithErrorOnIdCollision(
	resources resmap.ResMap) (err error) {
	ra.resMap, err = resmap.MergeWithErrorOnIdCollision(
		resources, ra.resMap)
	return err
}

func (ra *ResAccumulator) MergeResourcesWithOverride(
	resources resmap.ResMap) (err error) {
	ra.resMap, err = resmap.MergeWithOverride(
		ra.resMap, resources)
	return err
}

func (ra *ResAccumulator) MergeConfig(
	tConfig *config.TransformerConfig) (err error) {
	ra.tConfig, err = ra.tConfig.Merge(tConfig)
	return err
}

func (ra *ResAccumulator) MergeVars(incoming []types.Var) error {
	return ra.varSet.MergeSlice(incoming)
}

func (ra *ResAccumulator) MergeAccumulator(other *ResAccumulator) (err error) {
	err = ra.MergeResourcesWithErrorOnIdCollision(other.resMap)
	if err != nil {
		return err
	}
	err = ra.MergeConfig(other.tConfig)
	if err != nil {
		return err
	}
	return ra.varSet.MergeSet(&other.varSet)
}

// makeVarReplacementMap returns a map of Var names to
// their final values. The values are strings intended
// for substitution wherever the $(var.Name) occurs.
func (ra *ResAccumulator) makeVarReplacementMap() (map[string]string, error) {
	result := map[string]string{}
	for _, v := range ra.varSet.Set() {
		id := resid.NewResId(v.ObjRef.GVK(), v.ObjRef.Name)
		if r, found := ra.resMap.DemandOneMatchForId(id); found {
			s, err := r.GetFieldValue(v.FieldRef.FieldPath)
			if err != nil {
				return nil, fmt.Errorf("field path err for var: %+v", v)
			}
			result[v.Name] = s
		} else {
			// Should this be an error?
			log.Printf("var %v defined but not used", v)
		}
	}
	return result, nil
}

func (ra *ResAccumulator) Transform(t transformers.Transformer) error {
	return t.Transform(ra.resMap)
}

func (ra *ResAccumulator) ResolveVars() error {
	replacementMap, err := ra.makeVarReplacementMap()
	if err != nil {
		return err
	}
	return ra.Transform(transformers.NewRefVarTransformer(
		replacementMap, ra.tConfig.VarReference))
}

func (ra *ResAccumulator) FixBackReferences() (err error) {
	if ra.tConfig.NameReference == nil {
		return nil
	}
	return ra.Transform(transformers.NewNameReferenceTransformer(
		ra.tConfig.NameReference))
}
