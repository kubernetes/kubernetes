// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package target implements state for the set of all
// resources to customize.
package target

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/accumulator"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/ifc/transformer"
	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/plugins"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/plugin/builtin"
	"sigs.k8s.io/yaml"
)

// KustTarget encapsulates the entirety of a kustomization build.
type KustTarget struct {
	kustomization *types.Kustomization
	ldr           ifc.Loader
	rFactory      *resmap.Factory
	tFactory      transformer.Factory
	pLdr          *plugins.Loader
}

// NewKustTarget returns a new instance of KustTarget primed with a Loader.
func NewKustTarget(
	ldr ifc.Loader,
	rFactory *resmap.Factory,
	tFactory transformer.Factory,
	pLdr *plugins.Loader) (*KustTarget, error) {
	content, err := loadKustFile(ldr)
	if err != nil {
		return nil, err
	}
	content = types.FixKustomizationPreUnmarshalling(content)
	var k types.Kustomization
	err = unmarshal(content, &k)
	if err != nil {
		return nil, err
	}
	k.FixKustomizationPostUnmarshalling()
	errs := k.EnforceFields()
	if len(errs) > 0 {
		return nil, fmt.Errorf(
			"Failed to read kustomization file under %s:\n"+
				strings.Join(errs, "\n"), ldr.Root())
	}
	return &KustTarget{
		kustomization: &k,
		ldr:           ldr,
		rFactory:      rFactory,
		tFactory:      tFactory,
		pLdr:          pLdr,
	}, nil
}

func quoted(l []string) []string {
	r := make([]string, len(l))
	for i, v := range l {
		r[i] = "'" + v + "'"
	}
	return r
}

func commaOr(q []string) string {
	return strings.Join(q[:len(q)-1], ", ") + " or " + q[len(q)-1]
}

func loadKustFile(ldr ifc.Loader) ([]byte, error) {
	var content []byte
	match := 0
	for _, kf := range pgmconfig.KustomizationFileNames {
		c, err := ldr.Load(kf)
		if err == nil {
			match += 1
			content = c
		}
	}
	switch match {
	case 0:
		return nil, fmt.Errorf(
			"unable to find one of %v in directory '%s'",
			commaOr(quoted(pgmconfig.KustomizationFileNames)), ldr.Root())
	case 1:
		return content, nil
	default:
		return nil, fmt.Errorf(
			"Found multiple kustomization files under: %s\n", ldr.Root())
	}
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
	return kt.makeCustomizedResMap(types.GarbageIgnore)
}

func (kt *KustTarget) MakePruneConfigMap() (resmap.ResMap, error) {
	return kt.makeCustomizedResMap(types.GarbageCollect)
}

func (kt *KustTarget) makeCustomizedResMap(
	garbagePolicy types.GarbagePolicy) (resmap.ResMap, error) {
	ra, err := kt.AccumulateTarget()
	if err != nil {
		return nil, err
	}

	// The following steps must be done last, not as part of
	// the recursion implicit in AccumulateTarget.

	err = kt.addHashesToNames(ra)
	if err != nil {
		return nil, err
	}

	// Given that names have changed (prefixs/suffixes added),
	// fix all the back references to those names.
	err = ra.FixBackReferences()
	if err != nil {
		return nil, err
	}

	// With all the back references fixed, it's OK to resolve Vars.
	err = ra.ResolveVars()
	if err != nil {
		return nil, err
	}

	err = kt.computeInventory(ra, garbagePolicy)
	if err != nil {
		return nil, err
	}

	return ra.ResMap(), nil
}

func (kt *KustTarget) addHashesToNames(
	ra *accumulator.ResAccumulator) error {
	p := builtin.NewHashTransformerPlugin()
	err := kt.configureBuiltinPlugin(p, nil, "hash")
	if err != nil {
		return err
	}
	return ra.Transform(p)
}

func (kt *KustTarget) computeInventory(
	ra *accumulator.ResAccumulator, garbagePolicy types.GarbagePolicy) error {
	inv := kt.kustomization.Inventory
	if inv == nil {
		return nil
	}
	if inv.Type != "ConfigMap" {
		return fmt.Errorf("don't know how to do that")
	}

	if inv.ConfigMap.Namespace != kt.kustomization.Namespace {
		return fmt.Errorf("namespace mismatch")
	}

	p := builtin.NewInventoryTransformerPlugin()
	var c struct {
		Policy    string
		Name      string
		Namespace string
	}
	c.Name = inv.ConfigMap.Name
	c.Namespace = inv.ConfigMap.Namespace
	c.Policy = garbagePolicy.String()

	err := kt.configureBuiltinPlugin(p, c, "inventory")
	if err != nil {
		return err
	}
	return ra.Transform(p)
}

func (kt *KustTarget) shouldAddHashSuffixesToGeneratedResources() bool {
	return kt.kustomization.GeneratorOptions == nil ||
		!kt.kustomization.GeneratorOptions.DisableNameSuffixHash
}

// AccumulateTarget returns a new ResAccumulator,
// holding customized resources and the data/rules used
// to do so.  The name back references and vars are
// not yet fixed.
func (kt *KustTarget) AccumulateTarget() (
	ra *accumulator.ResAccumulator, err error) {
	ra = accumulator.MakeEmptyAccumulator()
	err = kt.accumulateResources(ra, kt.kustomization.Resources)
	if err != nil {
		return nil, errors.Wrap(err, "accumulating resources")
	}
	tConfig, err := config.MakeTransformerConfig(
		kt.ldr, kt.kustomization.Configurations)
	if err != nil {
		return nil, err
	}
	err = ra.MergeConfig(tConfig)
	if err != nil {
		return nil, errors.Wrapf(
			err, "merging config %v", tConfig)
	}
	crdTc, err := config.LoadConfigFromCRDs(kt.ldr, kt.kustomization.Crds)
	if err != nil {
		return nil, errors.Wrapf(
			err, "loading CRDs %v", kt.kustomization.Crds)
	}
	err = ra.MergeConfig(crdTc)
	if err != nil {
		return nil, errors.Wrapf(
			err, "merging CRDs %v", crdTc)
	}
	err = kt.runGenerators(ra)
	if err != nil {
		return nil, err
	}
	err = kt.runTransformers(ra)
	if err != nil {
		return nil, err
	}
	err = ra.MergeVars(kt.kustomization.Vars)
	if err != nil {
		return nil, errors.Wrapf(
			err, "merging vars %v", kt.kustomization.Vars)
	}
	return ra, nil
}

func (kt *KustTarget) runGenerators(
	ra *accumulator.ResAccumulator) error {
	generators, err := kt.configureBuiltinGenerators()
	if err != nil {
		return err
	}
	for _, g := range generators {
		resMap, err := g.Generate()
		if err != nil {
			return err
		}
		// The legacy generators allow override.
		err = ra.AbsorbAll(resMap)
		if err != nil {
			return errors.Wrapf(err, "merging from generator %v", g)
		}
	}
	generators, err = kt.configureExternalGenerators()
	if err != nil {
		return errors.Wrap(err, "loading generator plugins")
	}
	for _, g := range generators {
		resMap, err := g.Generate()
		if err != nil {
			return err
		}
		err = ra.AppendAll(resMap)
		if err != nil {
			return errors.Wrapf(err, "merging from generator %v", g)
		}
	}
	return nil
}

func (kt *KustTarget) configureExternalGenerators() ([]transformers.Generator, error) {
	ra := accumulator.MakeEmptyAccumulator()
	err := kt.accumulateResources(ra, kt.kustomization.Generators)
	if err != nil {
		return nil, err
	}
	return kt.pLdr.LoadGenerators(kt.ldr, ra.ResMap())
}

func (kt *KustTarget) runTransformers(ra *accumulator.ResAccumulator) error {
	patches, err := kt.rFactory.RF().SliceFromPatches(
		kt.ldr, kt.kustomization.PatchesStrategicMerge)
	if err != nil {
		return errors.Wrapf(
			err, "reading strategic merge patches %v",
			kt.kustomization.PatchesStrategicMerge)
	}
	var r []transformers.Transformer
	t, err := kt.tFactory.MakePatchTransformer(patches, kt.rFactory.RF())
	if err != nil {
		return err
	}
	r = append(r, t)
	tConfig := ra.GetTransformerConfig()
	lts, err := kt.configureBuiltinTransformers(tConfig)
	if err != nil {
		return err
	}
	r = append(r, lts...)
	lts, err = kt.configureExternalTransformers()
	if err != nil {
		return err
	}
	r = append(r, lts...)
	t = transformers.NewMultiTransformer(r)
	return ra.Transform(t)
}

func (kt *KustTarget) configureExternalTransformers() ([]transformers.Transformer, error) {
	ra := accumulator.MakeEmptyAccumulator()
	err := kt.accumulateResources(ra, kt.kustomization.Transformers)
	if err != nil {
		return nil, err
	}
	return kt.pLdr.LoadTransformers(kt.ldr, ra.ResMap())
}

// accumulateResources fills the given resourceAccumulator
// with resources read from the given list of paths.
func (kt *KustTarget) accumulateResources(
	ra *accumulator.ResAccumulator, paths []string) error {
	for _, path := range paths {
		ldr, err := kt.ldr.New(path)
		if err == nil {
			err = kt.accumulateDirectory(ra, ldr, path)
			if err != nil {
				return err
			}
		} else {
			err = kt.accumulateFile(ra, path)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (kt *KustTarget) accumulateDirectory(
	ra *accumulator.ResAccumulator, ldr ifc.Loader, path string) error {
	defer ldr.Cleanup()
	subKt, err := NewKustTarget(
		ldr, kt.rFactory, kt.tFactory, kt.pLdr)
	if err != nil {
		return errors.Wrapf(err, "couldn't make target for path '%s'", path)
	}
	subRa, err := subKt.AccumulateTarget()
	if err != nil {
		return errors.Wrapf(
			err, "recursed accumulation of path '%s'", path)
	}
	err = ra.MergeAccumulator(subRa)
	if err != nil {
		return errors.Wrapf(
			err, "recursed merging from path '%s'", path)
	}
	return nil
}

func (kt *KustTarget) accumulateFile(
	ra *accumulator.ResAccumulator, path string) error {
	resources, err := kt.rFactory.FromFile(kt.ldr, path)
	if err != nil {
		return errors.Wrapf(err, "accumulating resources from '%s'", path)
	}
	err = ra.AppendAll(resources)
	if err != nil {
		return errors.Wrapf(err, "merging resources from '%s'", path)
	}
	return nil
}
