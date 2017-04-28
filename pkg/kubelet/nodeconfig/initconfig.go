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

package nodeconfig

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"

	yaml "k8s.io/apimachinery/pkg/util/yaml"
	api "k8s.io/kubernetes/pkg/api"
	ccv1a1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
)

// initConfigExists is a simple existential check for the init config.
// If filesystem issues prevent it from determining existence, a fatal error is returned.
func (cc *NodeConfigController) initConfigExists() bool {
	ok, err := cc.dirExists(initConfigDir)
	if err != nil {
		fatalf("failed to determine whether init config exists, error: %v", err)
	}
	return ok
}

// loadInitConfig is a special loader for configuration in the init directory, as this
// consists of local configuration files rather than a checkpointed API object
func (cc *NodeConfigController) loadInitConfig() {
	const errfmt = "failed to load init config, error: %v"

	// if the node wasn't provisioned with an init config, indicate such by setting cc.initConfig to nil
	if !cc.initConfigExists() {
		cc.initConfig = nil
		return
	}
	infof("loading init config")

	dir := filepath.Join(cc.configDir, initConfigDir)
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		fatalf(errfmt, fmt.Errorf("failed to enumerate config files in dir %q, error: %v", dir, err))
	}

	bs := [][]byte{}
	for _, file := range files {
		p := filepath.Join(dir, file.Name())
		b, err := ioutil.ReadFile(p)
		if err != nil {
			fatalf("failed to read config file %q, error: %v", p, err)
		}
		bs = append(bs, b)
	}

	// no configuration is an error, some parameters are required
	if len(bs) == 0 {
		fatalf(errfmt, fmt.Errorf("no configuration files provided, but at least one is required if the %q dir exists", initConfigDir))
	}

	// TODO(mtaufen): Once the KubeletConfiguration type is decomposed (#44252), allow multiple files to be loaded here.
	data := bs[0]
	// no configuration is an error, some parameters are required
	if len(data) == 0 {
		fatalf(errfmt, fmt.Errorf("configuration was empty, but some parameters are required"))
	}

	// TODO(mtaufen): Once the KubeletConfiguration type is decomposed (#44252), extend this to allow a YAML stream in any given file.
	jdata, err := yaml.ToJSON(data)
	if err != nil {
		fatalf(errfmt, err)
	}
	kc := &ccv1a1.KubeletConfiguration{}
	if err := json.Unmarshal(jdata, kc); err != nil {
		fatalf(errfmt, err)
	}

	// run the defaulter on the loaded init config
	api.Scheme.Default(kc)

	cc.initConfig = kc
}

// validateInitConfig is a helper for validating the init config
func (cc *NodeConfigController) validateInitConfig() {
	if cc.initConfig == nil {
		infof("no init config provided, skipping validation of init config")
		return
	}

	infof("validating init config")
	if err := validateConfig(cc.initConfig); err != nil {
		fatalf("failed to validate the init config, error: %v")
	}
}
