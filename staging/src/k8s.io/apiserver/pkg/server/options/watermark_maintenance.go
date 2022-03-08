/*
Copyright 2022 The Kubernetes Authors.

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

type WatermarkMaintenanceOptions struct {
	Enable bool
}

func NewWatermarkMaintenanceOptions() *WatermarkMaintenanceOptions {
	return &WatermarkMaintenanceOptions{Enable: true}
}

func (o *WatermarkMaintenanceOptions) WithEnableOpt(enable bool) {
	o.Enable = enable
}

func (o *WatermarkMaintenanceOptions) Validate() []error {
	return nil
}

func (o *WatermarkMaintenanceOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.Enable, "enable-watermark-maintenance", o.Enable,
		"Set to false if you want to turn off priority-and-fairness and max-in-flight watermarks and it's maintenance")
}

func (o *WatermarkMaintenanceOptions) ApplyTo(c *server.Config) error {
	c.EnableWatermarkMaintenance = o.Enable
	return nil
}
