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

// Package target implements state for the set of all resources being customized.
package target

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resid"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/constants"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc/transformer"
	interror "sigs.k8s.io/kustomize/pkg/internal/error"
	patchtransformer "sigs.k8s.io/kustomize/pkg/patch/transformer"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/pkg/types"
)

// KustTarget encapsulates the entirety of a kustomization build.
type KustTarget struct {
	kustomization *types.Kustomization
	ldr           ifc.Loader
	fSys          fs.FileSystem
	rf            *resmap.Factory
	tcfg          *config.TransformerConfig
	ptf           transformer.Factory
}

// NewKustTarget returns a new instance of KustTarget primed with a Loader.
func NewKustTarget(
	ldr ifc.Loader, fSys fs.FileSystem,
	rf *resmap.Factory,
	ptf transformer.Factory,
	tcfg *config.TransformerConfig) (*KustTarget, error) {
	content, err := ldr.Load(constants.KustomizationFileName)
	if err != nil {
		return nil, err
	}

	var k types.Kustomization
	err = unmarshal(content, &k)
	if err != nil {
		return nil, err
	}
	k.DealWithDeprecatedFields()
	return &KustTarget{
		kustomization: &k,
		ldr:           ldr,
		fSys:          fSys,
		rf:            rf,
		tcfg:          tcfg,
		ptf:           ptf,
	}, nil
}

func unmarshal(y []byte, o interface{}) error {
	j, err := yaml.YAMLToJSON(y)
	if err != nil {
		return err
	}

	dec := json.NewDecoder(bytes.NewReader(j))
	dec.DisallowUnknownFields()
	return dec.Decode(o)
}

// MakeCustomizedResMap creates a ResMap per kustomization instructions.
// The Resources in the returned ResMap are fully customized.
func (kt *KustTarget) MakeCustomizedResMap() (resmap.ResMap, error) {
	m, err := kt.loadCustomizedResMap()
	if err != nil {
		return nil, err
	}
	return kt.resolveRefsToGeneratedResources(m)
}

// resolveRefsToGeneratedResources fixes all name references.
func (kt *KustTarget) resolveRefsToGeneratedResources(m resmap.ResMap) (resmap.ResMap, error) {
	if kt.kustomization.GeneratorOptions == nil || !kt.kustomization.GeneratorOptions.DisableHash {
		err := kt.ptf.MakeHashTransformer().Transform(m)
		if err != nil {
			return nil, err
		}
	}

	var r []transformers.Transformer
	t, err := transformers.NewNameReferenceTransformer(kt.tcfg.NameReference)
	if err != nil {
		return nil, err
	}
	r = append(r, t)

	refVars, err := kt.resolveRefVars(m)
	if err != nil {
		return nil, err
	}
	t = transformers.NewRefVarTransformer(refVars, kt.tcfg.VarReference)
	r = append(r, t)

	err = transformers.NewMultiTransformer(r).Transform(m)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// loadCustomizedResMap loads and customizes resources to build a ResMap.
func (kt *KustTarget) loadCustomizedResMap() (resmap.ResMap, error) {
	errs := &interror.KustomizationErrors{}
	result, err := kt.loadResMapFromBasesAndResources()
	if err != nil {
		errs.Append(errors.Wrap(err, "loadResMapFromBasesAndResources"))
	}
	crdTc, err := config.NewFactory(kt.ldr).LoadCRDs(kt.kustomization.Crds)
	kt.tcfg = kt.tcfg.Merge(crdTc)
	if err != nil {
		errs.Append(errors.Wrap(err, "LoadCRDs"))
	}
	resMap, err := kt.generateConfigMapsAndSecrets(errs)
	if err != nil {
		errs.Append(errors.Wrap(err, "generateConfigMapsAndSecrets"))
	}
	result, err = resmap.MergeWithOverride(result, resMap)
	if err != nil {
		return nil, err
	}

	patches, err := kt.rf.RF().SliceFromPatches(
		kt.ldr, kt.kustomization.PatchesStrategicMerge)
	if err != nil {
		errs.Append(errors.Wrap(err, "SliceFromPatches"))
	}

	if len(errs.Get()) > 0 {
		return nil, errs
	}

	var r []transformers.Transformer
	t, err := kt.newTransformer(patches)
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	t, err = patchtransformer.NewPatchJson6902Factory(kt.ldr).
		MakePatchJson6902Transformer(kt.kustomization.PatchesJson6902)
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	t, err = transformers.NewImageTagTransformer(kt.kustomization.ImageTags)
	if err != nil {
		return nil, err
	}
	r = append(r, t)

	err = transformers.NewMultiTransformer(r).Transform(result)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (kt *KustTarget) generateConfigMapsAndSecrets(
	errs *interror.KustomizationErrors) (resmap.ResMap, error) {
	kt.rf.Set(kt.fSys, kt.ldr)
	cms, err := kt.rf.NewResMapFromConfigMapArgs(kt.kustomization.ConfigMapGenerator, kt.kustomization.GeneratorOptions)
	if err != nil {
		errs.Append(errors.Wrap(err, "NewResMapFromConfigMapArgs"))
	}
	secrets, err := kt.rf.NewResMapFromSecretArgs(kt.kustomization.SecretGenerator, kt.kustomization.GeneratorOptions)
	if err != nil {
		errs.Append(errors.Wrap(err, "NewResMapFromSecretArgs"))
	}
	return resmap.MergeWithoutOverride(cms, secrets)
}

// Gets Bases and Resources as advertised.
func (kt *KustTarget) loadResMapFromBasesAndResources() (resmap.ResMap, error) {
	bases, errs := kt.loadCustomizedBases()
	resources, err := kt.rf.FromFiles(
		kt.ldr, kt.kustomization.Resources)
	if err != nil {
		errs.Append(errors.Wrap(err, "rawResources failed to read Resources"))
	}
	if len(errs.Get()) > 0 {
		return nil, errs
	}
	return resmap.MergeWithoutOverride(resources, bases)
}

// Loop through the Bases of this kustomization recursively loading resources.
// Combine into one ResMap, demanding unique Ids for each resource.
func (kt *KustTarget) loadCustomizedBases() (resmap.ResMap, *interror.KustomizationErrors) {
	var list []resmap.ResMap
	errs := &interror.KustomizationErrors{}
	for _, path := range kt.kustomization.Bases {
		ldr, err := kt.ldr.New(path)
		if err != nil {
			errs.Append(errors.Wrap(err, "couldn't make ldr for "+path))
			continue
		}
		target, err := NewKustTarget(
			ldr, kt.fSys, kt.rf, kt.ptf, kt.tcfg)
		if err != nil {
			errs.Append(errors.Wrap(err, "couldn't make target for "+path))
			continue
		}
		resMap, err := target.loadCustomizedResMap()
		if err != nil {
			errs.Append(errors.Wrap(err, "SemiResources"))
			continue
		}
		ldr.Cleanup()
		list = append(list, resMap)
	}
	result, err := resmap.MergeWithoutOverride(list...)
	if err != nil {
		errs.Append(errors.Wrap(err, "Merge failed"))
	}
	return result, errs
}

func (kt *KustTarget) loadBasesAsFlatList() ([]*KustTarget, error) {
	var result []*KustTarget
	errs := &interror.KustomizationErrors{}
	for _, path := range kt.kustomization.Bases {
		ldr, err := kt.ldr.New(path)
		if err != nil {
			errs.Append(err)
			continue
		}
		target, err := NewKustTarget(
			ldr, kt.fSys, kt.rf, kt.ptf, kt.tcfg)
		if err != nil {
			errs.Append(err)
			continue
		}
		result = append(result, target)
	}
	if len(errs.Get()) > 0 {
		return nil, errs
	}
	return result, nil
}

// newTransformer makes a Transformer that does everything except resolve generated names.
func (kt *KustTarget) newTransformer(patches []*resource.Resource) (transformers.Transformer, error) {
	var r []transformers.Transformer
	t, err := kt.ptf.MakePatchTransformer(patches, kt.rf.RF())
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	r = append(r, transformers.NewNamespaceTransformer(
		string(kt.kustomization.Namespace), kt.tcfg.NameSpace, kt.rf.RF()))
	t, err = transformers.NewNamePrefixTransformer(
		string(kt.kustomization.NamePrefix), kt.tcfg.NamePrefix)
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	t, err = transformers.NewLabelsMapTransformer(
		kt.kustomization.CommonLabels, kt.tcfg.CommonLabels)
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	t, err = transformers.NewAnnotationsMapTransformer(
		kt.kustomization.CommonAnnotations, kt.tcfg.CommonAnnotations)
	if err != nil {
		return nil, err
	}
	r = append(r, t)
	return transformers.NewMultiTransformer(r), nil
}

func (kt *KustTarget) resolveRefVars(m resmap.ResMap) (map[string]string, error) {
	result := map[string]string{}
	vars, err := kt.getAllVars()
	if err != nil {
		return result, err
	}
	for _, v := range vars {
		id := resid.NewResId(v.ObjRef.GVK(), v.ObjRef.Name)
		if r, found := m.DemandOneMatchForId(id); found {
			s, err := r.GetFieldValue(v.FieldRef.FieldPath)
			if err != nil {
				return nil, fmt.Errorf("failed to resolve referred var: %+v", v)
			}
			result[v.Name] = s
		} else {
			glog.Infof("couldn't resolve v: %v", v)
		}
	}
	return result, nil
}

// getAllVars returns all the "environment" style Var instances defined in the app.
func (kt *KustTarget) getAllVars() ([]types.Var, error) {
	var result []types.Var
	errs := &interror.KustomizationErrors{}

	bases, err := kt.loadBasesAsFlatList()
	if err != nil {
		return nil, err
	}

	// TODO: computing vars and resources for bases can be combined
	for _, b := range bases {
		vars, err := b.getAllVars()
		if err != nil {
			errs.Append(err)
			continue
		}
		b.ldr.Cleanup()
		result = append(result, vars...)
	}
	for _, v := range kt.kustomization.Vars {
		v.Defaulting()
		result = append(result, v)
	}
	if len(errs.Get()) > 0 {
		return nil, errs
	}
	return result, nil
}
