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
	"fmt"
	"os"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/server"
)

type FeatureOptions struct {
	EnableProfiling           bool
	EnableContentionProfiling bool
	EnableSwaggerUI           bool
	EnableLogsHandler         bool
	LogsDir                   string
}

func NewFeatureOptions() *FeatureOptions {
	defaults := server.NewConfig(serializer.CodecFactory{})

	return &FeatureOptions{
		EnableProfiling:           defaults.EnableProfiling,
		EnableContentionProfiling: defaults.EnableContentionProfiling,
		EnableSwaggerUI:           defaults.EnableSwaggerUI,
		EnableLogsHandler:         defaults.EnableLogsHandler,
	}
}

func (o *FeatureOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.EnableProfiling, "profiling", o.EnableProfiling,
		"Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&o.EnableContentionProfiling, "contention-profiling", o.EnableContentionProfiling,
		"Enable lock contention profiling, if profiling is enabled")
	fs.BoolVar(&o.EnableSwaggerUI, "enable-swagger-ui", o.EnableSwaggerUI,
		"Enables swagger ui on the apiserver at /swagger-ui")
	fs.BoolVar(&o.EnableLogsHandler, "enable-logs-handler", o.EnableLogsHandler,
		"If true, install a /logs handler for the apiserver logs.")
	fs.StringVar(&o.LogsDir, "logs-dir", o.LogsDir,
		"Logs directory used for serving logs handler. Default value is `/var/log`.")
}

func (o *FeatureOptions) ApplyTo(c *server.Config) error {
	if o == nil {
		return nil
	}

	c.EnableProfiling = o.EnableProfiling
	c.EnableContentionProfiling = o.EnableContentionProfiling
	c.EnableSwaggerUI = o.EnableSwaggerUI
	c.EnableLogsHandler = o.EnableLogsHandler
	c.LogsDir = o.LogsDir

	return nil
}

func (o *FeatureOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if o.EnableLogsHandler && len(o.LogsDir) > 0 {
		if exist, err := dirExists(o.LogsDir); !exist {
			errs = append(errs, fmt.Errorf("--logs-dir %q does not exist: %v", o.LogsDir, err))
		}
	}

	return errs
}

// dirExists checks if a path exists and is a directory.
func dirExists(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err == nil && fi.IsDir() {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
