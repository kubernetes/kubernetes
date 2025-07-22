// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package loader

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers"
	"sigs.k8s.io/kustomize/api/internal/plugins/execplugin"
	"sigs.k8s.io/kustomize/api/internal/plugins/fnplugin"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/kustomize/kyaml/resid"
)

// Loader loads plugins using a file loader (a different loader).
type Loader struct {
	pc *types.PluginConfig
	rf *resmap.Factory
	fs filesys.FileSystem

	// absolutePluginHome caches the location of a valid plugin root directory.
	// It should only be set once the directory's existence has been confirmed.
	absolutePluginHome string
}

func NewLoader(
	pc *types.PluginConfig, rf *resmap.Factory, fs filesys.FileSystem,
) *Loader {
	return &Loader{pc: pc, rf: rf, fs: fs}
}

// LoaderWithWorkingDir returns loader after setting its working directory.
// NOTE: This is not really a new loader since some of the Loader struct fields are pointers.
func (l *Loader) LoaderWithWorkingDir(wd string) *Loader {
	lpc := &types.PluginConfig{
		PluginRestrictions: l.pc.PluginRestrictions,
		BpLoadingOptions:   l.pc.BpLoadingOptions,
		FnpLoadingOptions:  l.pc.FnpLoadingOptions,
		HelmConfig:         l.pc.HelmConfig,
	}
	lpc.FnpLoadingOptions.WorkingDir = wd
	return &Loader{pc: lpc, rf: l.rf, fs: l.fs}
}

// Config provides the global (not plugin specific) PluginConfig data.
func (l *Loader) Config() *types.PluginConfig {
	return l.pc
}

func (l *Loader) LoadGenerators(
	ldr ifc.Loader, v ifc.Validator, rm resmap.ResMap) (
	result []*resmap.GeneratorWithProperties, err error,
) {
	for _, res := range rm.Resources() {
		g, err := l.LoadGenerator(ldr, v, res)
		if err != nil {
			return nil, fmt.Errorf("failed to load generator: %w", err)
		}
		generatorOrigin, err := resource.OriginFromCustomPlugin(res)
		if err != nil {
			return nil, fmt.Errorf("failed to get origin from CustomPlugin: %w", err)
		}
		result = append(result, &resmap.GeneratorWithProperties{Generator: g, Origin: generatorOrigin})
	}
	return result, nil
}

func (l *Loader) LoadGenerator(
	ldr ifc.Loader, v ifc.Validator, res *resource.Resource,
) (resmap.Generator, error) {
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
	ldr ifc.Loader, v ifc.Validator, rm resmap.ResMap,
) ([]*resmap.TransformerWithProperties, error) {
	var result []*resmap.TransformerWithProperties
	for _, res := range rm.Resources() {
		t, err := l.LoadTransformer(ldr, v, res)
		if err != nil {
			return nil, err
		}
		transformerOrigin, err := resource.OriginFromCustomPlugin(res)
		if err != nil {
			return nil, err
		}
		result = append(result, &resmap.TransformerWithProperties{Transformer: t, Origin: transformerOrigin})
	}
	return result, nil
}

func (l *Loader) LoadTransformer(
	ldr ifc.Loader, v ifc.Validator, res *resource.Resource,
) (*resmap.TransformerWithProperties, error) {
	c, err := l.loadAndConfigurePlugin(ldr, v, res)
	if err != nil {
		return nil, err
	}
	t, ok := c.(resmap.Transformer)
	if !ok {
		return nil, fmt.Errorf("plugin %s not a transformer", res.OrgId())
	}
	return &resmap.TransformerWithProperties{Transformer: t}, nil
}

func relativePluginPath(id resid.ResId) string {
	return filepath.Join(
		id.Group,
		id.Version,
		strings.ToLower(id.Kind))
}

func (l *Loader) AbsolutePluginPath(id resid.ResId) (string, error) {
	pluginHome, err := l.absPluginHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(pluginHome, relativePluginPath(id), id.Kind), nil
}

// absPluginHome is the home of kustomize Exec and Go plugins.
// Kustomize plugin configuration files are k8s-style objects
// containing the fields 'apiVersion' and 'kind', e.g.
//
//	apiVersion: apps/v1
//	kind: Deployment
//
// kustomize reads plugin configuration data from a file path
// specified in the 'generators:' or 'transformers:' field of a
// kustomization file.  For Exec and Go plugins, kustomize
// uses this data to both locate the plugin and configure it.
// Each Exec or Go plugin (its code, its tests, its supporting data
// files, etc.) must be housed in its own directory at
//
//	${absPluginHome}/${pluginApiVersion}/LOWERCASE(${pluginKind})
//
// where
//   - ${absPluginHome} is an absolute path, defined below.
//   - ${pluginApiVersion} is taken from the plugin config file.
//   - ${pluginKind} is taken from the plugin config file.
func (l *Loader) absPluginHome() (string, error) {
	// External plugins are disabled--return the dummy plugin root.
	if l.pc.PluginRestrictions != types.PluginRestrictionsNone {
		return konfig.NoPluginHomeSentinal, nil
	}
	// We've already determined plugin home--use the cached value.
	if l.absolutePluginHome != "" {
		return l.absolutePluginHome, nil
	}

	// Check default locations for a valid plugin root, and cache it if found.
	dir, err := konfig.DefaultAbsPluginHome(l.fs)
	if err != nil {
		return "", err
	}
	l.absolutePluginHome = dir
	return l.absolutePluginHome, nil
}

func isBuiltinPlugin(res *resource.Resource) bool {
	// TODO: the special string should appear in Group, not Version.
	return res.GetGvk().Group == "" &&
		res.GetGvk().Version == konfig.BuiltinPluginApiVersion
}

func (l *Loader) loadAndConfigurePlugin(
	ldr ifc.Loader,
	v ifc.Validator,
	res *resource.Resource,
) (c resmap.Configurable, err error) {
	if isBuiltinPlugin(res) {
		switch l.pc.BpLoadingOptions {
		case types.BploLoadFromFileSys:
			c, err = l.loadPlugin(res)
		case types.BploUseStaticallyLinked:
			// Instead of looking for and loading a .so file,
			// instantiate the plugin from a generated factory
			// function (see "pluginator").  Being able to do this
			// is what makes a plugin "builtin".
			c, err = l.makeBuiltinPlugin(res.GetGvk())
		default:
			err = fmt.Errorf(
				"unknown plugin loader behavior specified: %s %v", res.GetGvk().String(),
				l.pc.BpLoadingOptions)
		}
	} else {
		switch l.pc.PluginRestrictions {
		case types.PluginRestrictionsNone:
			c, err = l.loadPlugin(res)
		case types.PluginRestrictionsBuiltinsOnly:
			err = types.NewErrOnlyBuiltinPluginsAllowed(res.OrgId().Kind)
		default:
			err = fmt.Errorf(
				"unknown plugin restriction specified: %v",
				l.pc.PluginRestrictions)
		}
	}
	if err != nil {
		return nil, err
	}
	yaml, err := res.AsYAML()
	if err != nil {
		return nil, errors.WrapPrefixf(err, "marshalling yaml from res %s", res.OrgId())
	}
	err = c.Config(resmap.NewPluginHelpers(ldr, v, l.rf, l.pc), yaml)
	if err != nil {
		return nil, errors.WrapPrefixf(
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

func (l *Loader) loadPlugin(res *resource.Resource) (resmap.Configurable, error) {
	spec, err := fnplugin.GetFunctionSpec(res)
	if err != nil {
		return nil, fmt.Errorf("loader: %w", err)
	}
	if spec != nil {
		// validation check that function mounts are under the current kustomization directory
		for _, mount := range spec.Container.StorageMounts {
			if filepath.IsAbs(mount.Src) {
				return nil, errors.Errorf("plugin %s with mount path '%s' is not permitted; "+
					"mount paths must be relative to the current kustomization directory", res.OrgId(), mount.Src)
			}
			if strings.HasPrefix(filepath.Clean(mount.Src), "../") {
				return nil, errors.Errorf("plugin %s with mount path '%s' is not permitted; "+
					"mount paths must be under the current kustomization directory", res.OrgId(), mount.Src)
			}
		}
		return fnplugin.NewFnPlugin(&l.pc.FnpLoadingOptions), nil
	}
	return l.loadExecOrGoPlugin(res.OrgId())
}

func (l *Loader) loadExecOrGoPlugin(resId resid.ResId) (resmap.Configurable, error) {
	absPluginPath, err := l.AbsolutePluginPath(resId)
	if err != nil {
		return nil, err
	}
	// First try to load the plugin as an executable.
	p := execplugin.NewExecPlugin(absPluginPath)
	if err = p.ErrIfNotExecutable(); err == nil {
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
	c, err := l.loadGoPlugin(resId, absPluginPath+".so")
	if err != nil {
		return nil, err
	}
	return c, nil
}
