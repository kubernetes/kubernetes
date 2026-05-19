// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package execplugin

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"sigs.k8s.io/kustomize/api/internal/plugins/utils"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/yaml"
)

const (
	tmpConfigFilePrefix = "kust-plugin-config-"
	maxArgStringLength  = 131071
)

// ExecPlugin record the name and args of an executable
// It triggers the executable generator and transformer
type ExecPlugin struct {
	// absolute path of the executable
	path string

	// Optional command line arguments to the executable
	// pulled from specially named fields in cfg.
	// This is for executables that don't want to parse YAML.
	args []string

	// Plugin configuration data.
	cfg []byte

	// PluginHelpers
	h *resmap.PluginHelpers
}

func NewExecPlugin(p string) *ExecPlugin {
	return &ExecPlugin{path: p}
}

func (p *ExecPlugin) ErrIfNotExecutable() error {
	f, err := os.Stat(p.path)
	if err != nil {
		return err
	}
	// In Windows, it is not possible to determine whether a
	// file is executable through file mode.
	// TODO: provide for setting the executable FileMode bit on Windows
	// The (fs *fileStat) Mode() (m FileMode) {} function in
	// https://golang.org/src/os/types_windows.go
	// lacks the ability to set the FileMode executable bit in response
	// to file data on Windows.
	if f.Mode()&0111 == 0000 && runtime.GOOS != "windows" {
		return fmt.Errorf("unexecutable plugin at: %s", p.path)
	}
	return nil
}

func (p *ExecPlugin) Path() string {
	return p.path
}

func (p *ExecPlugin) Args() []string {
	return p.args
}

func (p *ExecPlugin) Cfg() []byte {
	return p.cfg
}

func (p *ExecPlugin) Config(h *resmap.PluginHelpers, config []byte) error {
	p.h = h
	p.cfg = config
	return p.processOptionalArgsFields()
}

type argsConfig struct {
	ArgsOneLiner string `json:"argsOneLiner,omitempty" yaml:"argsOneLiner,omitempty"`
	ArgsFromFile string `json:"argsFromFile,omitempty" yaml:"argsFromFile,omitempty"`
}

func (p *ExecPlugin) processOptionalArgsFields() error {
	var c argsConfig
	err := yaml.Unmarshal(p.cfg, &c)
	if err != nil {
		return err
	}
	if c.ArgsOneLiner != "" {
		argsTolenSlice, err := ShlexSplit(c.ArgsOneLiner)
		if err != nil {
			return fmt.Errorf("failed to parse argsOneLiner: %w", err)
		}
		p.args = argsTolenSlice
	}
	if c.ArgsFromFile != "" {
		content, err := p.h.Loader().Load(c.ArgsFromFile)
		if err != nil {
			return err
		}
		for _, x := range strings.Split(string(content), "\n") {
			x := strings.TrimLeft(x, " ")
			if x != "" {
				p.args = append(p.args, x)
			}
		}
	}
	return nil
}

func (p *ExecPlugin) Generate() (resmap.ResMap, error) {
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

func (p *ExecPlugin) Transform(rm resmap.ResMap) error {
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
	return utils.UpdateResMapValues(p.path, p.h, output, rm)
}

// invokePlugin writes plugin config to a temp file, then
// passes the full temp file path as the first arg to a process
// running the plugin binary.  Process output is returned.
func (p *ExecPlugin) invokePlugin(input []byte) ([]byte, error) {
	f, err := os.CreateTemp("", tmpConfigFilePrefix)
	if err != nil {
		return nil, errors.WrapPrefixf(
			err, "creating tmp plugin config file")
	}
	_, err = f.Write(p.cfg)
	if err != nil {
		return nil, errors.WrapPrefixf(
			err, "writing plugin config to "+f.Name())
	}
	err = f.Close()
	if err != nil {
		return nil, errors.WrapPrefixf(
			err, "closing plugin config file "+f.Name())
	}
	//nolint:gosec
	cmd := exec.Command(
		p.path, append([]string{f.Name()}, p.args...)...)
	cmd.Env = p.getEnv()
	cmd.Stdin = bytes.NewReader(input)
	var stdErr bytes.Buffer
	cmd.Stderr = &stdErr
	if _, err := os.Stat(p.h.Loader().Root()); err == nil {
		cmd.Dir = p.h.Loader().Root()
	}
	result, err := cmd.Output()
	if err != nil {
		//nolint:govet
		return nil, errors.WrapPrefixf(
			fmt.Errorf("failure in plugin configured via %s; %w",
				f.Name(), err), stdErr.String())
	}
	return result, os.Remove(f.Name())
}

func (p *ExecPlugin) getEnv() []string {
	env := os.Environ()
	pluginConfigString := "KUSTOMIZE_PLUGIN_CONFIG_STRING=" + string(p.cfg)
	if len(pluginConfigString) <= maxArgStringLength {
		env = append(env, pluginConfigString)
	} else {
		log.Printf("KUSTOMIZE_PLUGIN_CONFIG_STRING exceeds hard limit of %d characters, the environment variable "+
			"will be omitted", maxArgStringLength)
	}
	pluginConfigRoot := "KUSTOMIZE_PLUGIN_CONFIG_ROOT=" + p.h.Loader().Root()
	if len(pluginConfigRoot) <= maxArgStringLength {
		env = append(env, pluginConfigRoot)
	} else {
		log.Printf("KUSTOMIZE_PLUGIN_CONFIG_ROOT exceeds hard limit of %d characters, the environment variable "+
			"will be omitted", maxArgStringLength)
	}
	return env
}
