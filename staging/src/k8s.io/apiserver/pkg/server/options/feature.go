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

package options

import (
	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/server"
)

type FeatureOptions struct {
	EnableProfiling           bool
	EnableContentionProfiling bool
	EnableSwaggerUI           bool
}

func NewFeatureOptions() *FeatureOptions {
	defaults := server.NewConfig()

	return &FeatureOptions{
		EnableProfiling:           defaults.EnableProfiling,
		EnableContentionProfiling: defaults.EnableContentionProfiling,
		EnableSwaggerUI:           defaults.EnableSwaggerUI,
	}
}

func (o *FeatureOptions) AddFlags(fs *pflag.FlagSet) {
	fs.BoolVar(&o.EnableProfiling, "profiling", o.EnableProfiling,
		"Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&o.EnableContentionProfiling, "contention-profiling", o.EnableContentionProfiling,
		"Enable lock contention profiling, if profiling is enabled")
	fs.BoolVar(&o.EnableSwaggerUI, "enable-swagger-ui", o.EnableSwaggerUI,
		"Enables swagger ui on the apiserver at /swagger-ui")
}

func (o *FeatureOptions) ApplyTo(c *server.Config) error {
	c.EnableProfiling = o.EnableProfiling
	c.EnableContentionProfiling = o.EnableContentionProfiling
	c.EnableSwaggerUI = o.EnableSwaggerUI

	return nil
}
