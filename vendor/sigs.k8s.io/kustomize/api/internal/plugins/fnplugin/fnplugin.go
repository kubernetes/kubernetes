// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package fnplugin

import (
	"bytes"
	"fmt"

	"github.com/pkg/errors"

	"sigs.k8s.io/kustomize/api/internal/plugins/utils"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/runtimeutil"
	"sigs.k8s.io/kustomize/kyaml/runfn"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// FnPlugin is the struct to hold function information
type FnPlugin struct {
	// Function runner
	runFns runfn.RunFns

	// Plugin configuration data.
	cfg []byte

	// Plugin name cache for error output
	pluginName string

	// PluginHelpers
	h *resmap.PluginHelpers
}

func bytesToRNode(yml []byte) (*yaml.RNode, error) {
	rnode, err := yaml.Parse(string(yml))
	if err != nil {
		return nil, err
	}
	return rnode, nil
}

func resourceToRNode(res *resource.Resource) (*yaml.RNode, error) {
	yml, err := res.AsYAML()
	if err != nil {
		return nil, err
	}

	return bytesToRNode(yml)
}

// GetFunctionSpec return function spec is there is. Otherwise return nil
func GetFunctionSpec(res *resource.Resource) *runtimeutil.FunctionSpec {
	rnode, err := resourceToRNode(res)
	if err != nil {
		return nil
	}

	return runtimeutil.GetFunctionSpec(rnode)
}

func toStorageMounts(mounts []string) []runtimeutil.StorageMount {
	var sms []runtimeutil.StorageMount
	for _, mount := range mounts {
		sms = append(sms, runtimeutil.StringToStorageMount(mount))
	}
	return sms
}

// NewFnPlugin creates a FnPlugin struct
func NewFnPlugin(o *types.FnPluginLoadingOptions) *FnPlugin {
	return &FnPlugin{
		runFns: runfn.RunFns{
			Functions:      []*yaml.RNode{},
			Network:        o.Network,
			EnableStarlark: o.EnableStar,
			EnableExec:     o.EnableExec,
			StorageMounts:  toStorageMounts(o.Mounts),
			Env:            o.Env,
		},
	}
}

// Cfg returns function config
func (p *FnPlugin) Cfg() []byte {
	return p.cfg
}

// Config is called by kustomize to pass-in config information
func (p *FnPlugin) Config(h *resmap.PluginHelpers, config []byte) error {
	p.h = h
	p.cfg = config

	fn, err := bytesToRNode(p.cfg)
	if err != nil {
		return err
	}

	meta, err := fn.GetMeta()
	if err != nil {
		return err
	}

	p.pluginName = fmt.Sprintf("api: %s, kind: %s, name: %s",
		meta.APIVersion, meta.Kind, meta.Name)

	return nil
}

// Generate is called when run as generator
func (p *FnPlugin) Generate() (resmap.ResMap, error) {
	output, err := p.invokePlugin(nil)
	if err != nil {
		return nil, err
	}
	rm, err := p.h.ResmapFactory().NewResMapFromBytes(output)
	if err != nil {
		return nil, err
	}
	return utils.UpdateResourceOptions(rm)
}

// Transform is called when run as transformer
func (p *FnPlugin) Transform(rm resmap.ResMap) error {
	// add ResIds as annotations to all objects so that we can add them back
	inputRM, err := utils.GetResMapWithIDAnnotation(rm)
	if err != nil {
		return err
	}

	// encode the ResMap so it can be fed to the plugin
	resources, err := inputRM.AsYaml()
	if err != nil {
		return err
	}

	// invoke the plugin with resources as the input
	output, err := p.invokePlugin(resources)
	if err != nil {
		return fmt.Errorf("%v %s", err, string(output))
	}

	// update the original ResMap based on the output
	return utils.UpdateResMapValues(p.pluginName, p.h, output, rm)
}

func injectAnnotation(input *yaml.RNode, k, v string) error {
	err := input.PipeE(yaml.SetAnnotation(k, v))
	if err != nil {
		return err
	}
	return nil
}

// invokePlugin uses Function runner to run function as plugin
func (p *FnPlugin) invokePlugin(input []byte) ([]byte, error) {
	// get function config rnode
	functionConfig, err := bytesToRNode(p.cfg)
	if err != nil {
		return nil, err
	}

	// This annotation will let kustomize ingnore this item in output
	err = injectAnnotation(functionConfig, "config.kubernetes.io/local-config", "true")
	if err != nil {
		return nil, err
	}
	// we need to add config as input for generators. Some of them don't work with FunctionConfig
	// and in addition kio.Pipeline won't create anything if there are no objects
	// see https://github.com/kubernetes-sigs/kustomize/blob/master/kyaml/kio/kio.go#L93
	// Since we added `local-config` annotation so it will be ignored in generator output
	// TODO(donnyxia): This is actually not used by generator and only used to bypass a kio limitation.
	// Need better solution.
	if input == nil {
		yml, err := functionConfig.String()
		if err != nil {
			return nil, err
		}
		input = []byte(yml)
	}

	// Configure and Execute Fn. We don't need to convert resources to ResourceList here
	// because function runtime will do that. See kyaml/fn/runtime/runtimeutil/runtimeutil.go
	var ouputBuffer bytes.Buffer
	p.runFns.Input = bytes.NewReader(input)
	p.runFns.Functions = append(p.runFns.Functions, functionConfig)
	p.runFns.Output = &ouputBuffer

	err = p.runFns.Execute()
	if err != nil {
		return nil, errors.Wrap(
			err, "couldn't execute function")
	}

	return ouputBuffer.Bytes(), nil
}
