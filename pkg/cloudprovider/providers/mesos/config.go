/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package mesos

import (
	"io"
	"time"

	"code.google.com/p/gcfg"
)

const (
	DefaultMesosMaster       = "localhost:5050"
	DefaultHttpClientTimeout = time.Duration(10) * time.Second
	DefaultStateCacheTTL     = time.Duration(5) * time.Second
)

// Example Mesos cloud provider configuration file:
//
// [mesos-cloud]
//  mesos-master        = leader.mesos:5050
//	http-client-timeout = 500ms
//	state-cache-ttl     = 1h

type ConfigWrapper struct {
	Mesos_Cloud Config
}

type Config struct {
	MesosMaster            string   `gcfg:"mesos-master"`
	MesosHttpClientTimeout Duration `gcfg:"http-client-timeout"`
	StateCacheTTL          Duration `gcfg:"state-cache-ttl"`
}

type Duration struct {
	Duration time.Duration `gcfg:"duration"`
}

func (d *Duration) UnmarshalText(data []byte) error {
	underlying, err := time.ParseDuration(string(data))
	if err == nil {
		d.Duration = underlying
	}
	return err
}

func createDefaultConfig() *Config {
	return &Config{
		MesosMaster:            DefaultMesosMaster,
		MesosHttpClientTimeout: Duration{Duration: DefaultHttpClientTimeout},
		StateCacheTTL:          Duration{Duration: DefaultStateCacheTTL},
	}
}

func readConfig(configReader io.Reader) (*Config, error) {
	config := createDefaultConfig()
	wrapper := &ConfigWrapper{Mesos_Cloud: *config}
	if configReader != nil {
		if err := gcfg.ReadInto(wrapper, configReader); err != nil {
			return nil, err
		}
		config = &(wrapper.Mesos_Cloud)
	}
	return config, nil
}
