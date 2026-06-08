// Copyright 2024 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0
//go:build !kustomize_disable_go_plugin_support

package loader

import (
	"fmt"
	"log"
	"plugin"
	"reflect"

	"sigs.k8s.io/kustomize/api/internal/plugins/utils"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/resid"
)

// registry is a means to avoid trying to load the same .so file
// into memory more than once, which results in an error.
// Each test makes its own loader, and tries to load its own plugins,
// but the loaded .so files are in shared memory, so one will get
// "this plugin already loaded" errors if the registry is maintained
// as a Loader instance variable.  So make it a package variable.
var registry = make(map[string]resmap.Configurable) //nolint:gochecknoglobals

func copyPlugin(c resmap.Configurable) resmap.Configurable {
	indirect := reflect.Indirect(reflect.ValueOf(c))
	newIndirect := reflect.New(indirect.Type())
	newIndirect.Elem().Set(reflect.ValueOf(indirect.Interface()))
	newNamed := newIndirect.Interface()
	return newNamed.(resmap.Configurable) //nolint:forcetypeassert
}

func (l *Loader) loadGoPlugin(id resid.ResId, absPath string) (resmap.Configurable, error) {
	regId := relativePluginPath(id)
	if c, ok := registry[regId]; ok {
		return copyPlugin(c), nil
	}
	if !utils.FileExists(absPath) {
		return nil, fmt.Errorf(
			"expected file with Go object code at: %s", absPath)
	}
	log.Printf("Attempting plugin load from %q", absPath)
	p, err := plugin.Open(absPath)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "plugin %s fails to load", absPath)
	}
	symbol, err := p.Lookup(konfig.PluginSymbol)
	if err != nil {
		return nil, errors.WrapPrefixf(
			err, "plugin %s doesn't have symbol %s",
			regId, konfig.PluginSymbol)
	}
	c, ok := symbol.(resmap.Configurable)
	if !ok {
		return nil, fmt.Errorf("plugin %q not configurable", regId)
	}
	registry[regId] = c
	return copyPlugin(c), nil
}
