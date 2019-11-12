// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package loader

import (
	"fmt"
	"os"
	"path/filepath"
	"plugin"
	"reflect"
	"strings"

	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers"
	"sigs.k8s.io/kustomize/api/internal/plugins/execplugin"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
)

type Loader struct {
	pc *types.PluginConfig
	rf *resmap.Factory
}

func NewLoader(
	pc *types.PluginConfig, rf *resmap.Factory) *Loader {
	return &Loader{pc: pc, rf: rf}
}

func (l *Loader) LoadGenerators(
	ldr ifc.Loader, v ifc.Validator, rm resmap.ResMap) ([]resmap.Generator, error) {
	var result []resmap.Generator
	for _, res := range rm.Resources() {
		g, err := l.LoadGenerator(ldr, v, res)
		if err != nil {
			return nil, err
		}
		result = append(result, g)
	}
	return result, nil
}

func (l *Loader) LoadGenerator(
	ldr ifc.Loader, v ifc.Validator, res *resource.Resource) (resmap.Generator, error) {
	c, err := l.loadAndConfigurePlugin(ldr, v, res)
	if err != nil {
		return nil, err
	}
	g, ok := c.(resmap.Generator)
	if !ok {
		return nil, fmt.Errorf("plugin %s not a generator", res.OrgId())
	}
	return g, nil
}

func (l *Loader) LoadTransformers(
	ldr ifc.Loader, v ifc.Validator, rm resmap.ResMap) ([]resmap.Transformer, error) {
	var result []resmap.Transformer
	for _, res := range rm.Resources() {
		t, err := l.LoadTransformer(ldr, v, res)
		if err != nil {
			return nil, err
		}
		result = append(result, t)
	}
	return result, nil
}

func (l *Loader) LoadTransformer(
	ldr ifc.Loader, v ifc.Validator, res *resource.Resource) (resmap.Transformer, error) {
	c, err := l.loadAndConfigurePlugin(ldr, v, res)
	if err != nil {
		return nil, err
	}
	t, ok := c.(resmap.Transformer)
	if !ok {
		return nil, fmt.Errorf("plugin %s not a transformer", res.OrgId())
	}
	return t, nil
}

func relativePluginPath(id resid.ResId) string {
	return filepath.Join(
		id.Group,
		id.Version,
		strings.ToLower(id.Kind))
}

func AbsolutePluginPath(pc *types.PluginConfig, id resid.ResId) string {
	return filepath.Join(
		pc.AbsPluginHome, relativePluginPath(id), id.Kind)
}

func (l *Loader) absolutePluginPath(id resid.ResId) string {
	return AbsolutePluginPath(l.pc, id)
}

func isBuiltinPlugin(res *resource.Resource) bool {
	// TODO: the special string should appear in Group, not Version.
	return res.GetGvk().Group == "" &&
		res.GetGvk().Version == konfig.BuiltinPluginApiVersion
}

func (l *Loader) loadAndConfigurePlugin(
	ldr ifc.Loader, v ifc.Validator, res *resource.Resource) (c resmap.Configurable, err error) {
	if isBuiltinPlugin(res) {
		// Instead of looking for and loading a .so file, just
		// instantiate the plugin from a generated factory
		// function (see "pluginator").  Being able to do this
		// is what makes a plugin "builtin".
		c, err = l.makeBuiltinPlugin(res.GetGvk())
	} else if l.pc.PluginRestrictions == types.PluginRestrictionsNone {
		c, err = l.loadPlugin(res.OrgId())
	} else {
		err = types.NewErrOnlyBuiltinPluginsAllowed(res.OrgId().Kind)
	}
	if err != nil {
		return nil, err
	}
	yaml, err := res.AsYAML()
	if err != nil {
		return nil, errors.Wrapf(err, "marshalling yaml from res %s", res.OrgId())
	}
	err = c.Config(resmap.NewPluginHelpers(ldr, v, l.rf), yaml)
	if err != nil {
		return nil, errors.Wrapf(
			err, "plugin %s fails configuration", res.OrgId())
	}
	return c, nil
}

func (l *Loader) makeBuiltinPlugin(r resid.Gvk) (resmap.Configurable, error) {
	bpt := builtinhelpers.GetBuiltinPluginType(r.Kind)
	if f, ok := builtinhelpers.GeneratorFactories[bpt]; ok {
		return f(), nil
	}
	if f, ok := builtinhelpers.TransformerFactories[bpt]; ok {
		return f(), nil
	}
	return nil, errors.Errorf("unable to load builtin %s", r)
}

func (l *Loader) loadPlugin(resId resid.ResId) (resmap.Configurable, error) {
	// First try to load the plugin as an executable.
	p := execplugin.NewExecPlugin(l.absolutePluginPath(resId))
	err := p.ErrIfNotExecutable()
	if err == nil {
		return p, nil
	}
	if !os.IsNotExist(err) {
		// The file exists, but something else is wrong,
		// likely it's not executable.
		// Assume the user forgot to set the exec bit,
		// and return an error, rather than adding ".so"
		// to the name and attempting to load it as a Go
		// plugin, which will likely fail and result
		// in an obscure message.
		return nil, err
	}
	// Failing the above, try loading it as a Go plugin.
	c, err := l.loadGoPlugin(resId)
	if err != nil {
		return nil, err
	}
	return c, nil
}

// registry is a means to avoid trying to load the same .so file
// into memory more than once, which results in an error.
// Each test makes its own loader, and tries to load its own plugins,
// but the loaded .so files are in shared memory, so one will get
// "this plugin already loaded" errors if the registry is maintained
// as a Loader instance variable.  So make it a package variable.
var registry = make(map[string]resmap.Configurable)

func (l *Loader) loadGoPlugin(id resid.ResId) (resmap.Configurable, error) {
	regId := relativePluginPath(id)
	if c, ok := registry[regId]; ok {
		return copyPlugin(c), nil
	}
	absPath := l.absolutePluginPath(id)
	p, err := plugin.Open(absPath + ".so")
	if err != nil {
		return nil, errors.Wrapf(err, "plugin %s fails to load", absPath)
	}
	symbol, err := p.Lookup(konfig.PluginSymbol)
	if err != nil {
		return nil, errors.Wrapf(
			err, "plugin %s doesn't have symbol %s",
			regId, konfig.PluginSymbol)
	}
	c, ok := symbol.(resmap.Configurable)
	if !ok {
		return nil, fmt.Errorf("plugin '%s' not configurable", regId)
	}
	registry[regId] = c
	return copyPlugin(c), nil
}

func copyPlugin(c resmap.Configurable) resmap.Configurable {
	indirect := reflect.Indirect(reflect.ValueOf(c))
	newIndirect := reflect.New(indirect.Type())
	newIndirect.Elem().Set(reflect.ValueOf(indirect.Interface()))
	newNamed := newIndirect.Interface()
	return newNamed.(resmap.Configurable)
}
