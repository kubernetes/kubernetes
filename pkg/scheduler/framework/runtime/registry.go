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

package runtime

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
	fwk "k8s.io/kube-scheduler/framework"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"sigs.k8s.io/yaml"
)

// PluginFactory is a function that builds a plugin.
type PluginFactory = func(ctx context.Context, configuration runtime.Object, f fwk.Handle) (fwk.Plugin, error)

// PluginFactoryWithFts is a function that builds a plugin with certain feature gates.
type PluginFactoryWithFts[T fwk.Plugin] func(context.Context, runtime.Object, fwk.Handle, plfeature.Features) (T, error)

// FactoryAdapter can be used to inject feature gates for a plugin that needs
// them when the caller expects the older PluginFactory method.
func FactoryAdapter[T fwk.Plugin](fts plfeature.Features, withFts PluginFactoryWithFts[T]) PluginFactory {
	return func(ctx context.Context, plArgs runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return withFts(ctx, plArgs, fh, fts)
	}
}

// DecodeInto decodes configuration whose type is *runtime.Unknown to the interface into.
func DecodeInto(obj runtime.Object, into interface{}) error {
	if obj == nil {
		return nil
	}
	configuration, ok := obj.(*runtime.Unknown)
	if !ok {
		return fmt.Errorf("want args of type runtime.Unknown, got %T", obj)
	}
	if configuration.Raw == nil {
		return nil
	}

	switch configuration.ContentType {
	// If ContentType is empty, it means ContentTypeJSON by default.
	case runtime.ContentTypeJSON, "":
		return json.Unmarshal(configuration.Raw, into)
	case runtime.ContentTypeYAML:
		return yaml.Unmarshal(configuration.Raw, into)
	default:
		return fmt.Errorf("not supported content type %s", configuration.ContentType)
	}
}

// Registry is a collection of all available plugins. The framework uses a
// registry to enable and initialize configured plugins.
// All plugins must be in the registry before initializing the framework.
type Registry map[string]PluginFactory

// Register adds a new plugin to the registry. If a plugin with the same name
// exists, it returns an error.
func (r Registry) Register(name string, factory PluginFactory) error {
	if _, ok := r[name]; ok {
		return fmt.Errorf("a plugin named %v already exists", name)
	}
	r[name] = factory
	return nil
}

// Unregister removes an existing plugin from the registry. If no plugin with
// the provided name exists, it returns an error.
func (r Registry) Unregister(name string) error {
	if _, ok := r[name]; !ok {
		return fmt.Errorf("no plugin named %v exists", name)
	}
	delete(r, name)
	return nil
}

// Merge merges the provided registry to the current one.
func (r Registry) Merge(in Registry) error {
	for name, factory := range in {
		if err := r.Register(name, factory); err != nil {
			return err
		}
	}
	return nil
}
