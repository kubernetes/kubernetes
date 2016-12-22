/*
Copyright 2016 The Kubernetes Authors.

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

package config

// nopSync does no synchronization, used when the DNS server is
// started without a ConfigMap configured.
type nopSync struct {
	config *Config
}

var _ Sync = (*nopSync)(nil)

func NewNopSync(config *Config) Sync {
	return &nopSync{config: config}
}

func (sync *nopSync) Once() (*Config, error) {
	return sync.config, nil
}

func (sync *nopSync) Periodic() <-chan *Config {
	return make(chan *Config)
}
