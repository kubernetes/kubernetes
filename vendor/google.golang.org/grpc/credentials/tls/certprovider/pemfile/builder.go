/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package pemfile

import (
	"encoding/json"
	"fmt"
	"time"

	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/durationpb"
)

const (
	// PluginName is the name of the PEM file watcher plugin.
	PluginName             = "file_watcher"
	defaultRefreshInterval = 10 * time.Minute
)

func init() {
	certprovider.Register(&pluginBuilder{})
}

type pluginBuilder struct{}

func (p *pluginBuilder) ParseConfig(c any) (*certprovider.BuildableConfig, error) {
	data, ok := c.(json.RawMessage)
	if !ok {
		return nil, fmt.Errorf("meshca: unsupported config type: %T", c)
	}
	opts, err := pluginConfigFromJSON(data)
	if err != nil {
		return nil, err
	}
	return certprovider.NewBuildableConfig(PluginName, opts.canonical(), func(certprovider.BuildOptions) certprovider.Provider {
		return newProvider(opts)
	}), nil
}

func (p *pluginBuilder) Name() string {
	return PluginName
}

func pluginConfigFromJSON(jd json.RawMessage) (Options, error) {
	// The only difference between this anonymous struct and the Options struct
	// is that the refresh_interval is represented here as a duration proto,
	// while in the latter a time.Duration is used.
	cfg := &struct {
		CertificateFile          string          `json:"certificate_file,omitempty"`
		PrivateKeyFile           string          `json:"private_key_file,omitempty"`
		CACertificateFile        string          `json:"ca_certificate_file,omitempty"`
		SPIFFETrustBundleMapFile string          `json:"spiffe_trust_bundle_map_file,omitempty"`
		RefreshInterval          json.RawMessage `json:"refresh_interval,omitempty"`
	}{}
	if err := json.Unmarshal(jd, cfg); err != nil {
		return Options{}, fmt.Errorf("pemfile: json.Unmarshal(%s) failed: %v", string(jd), err)
	}
	if !envconfig.XDSSPIFFEEnabled {
		cfg.SPIFFETrustBundleMapFile = ""
	}

	opts := Options{
		CertFile:            cfg.CertificateFile,
		KeyFile:             cfg.PrivateKeyFile,
		RootFile:            cfg.CACertificateFile,
		SPIFFEBundleMapFile: cfg.SPIFFETrustBundleMapFile,
		// Refresh interval is the only field in the configuration for which we
		// support a default value. We cannot possibly have valid defaults for
		// file paths to watch. Also, it is valid to specify an empty path for
		// some of those fields if the user does not want to watch them.
		RefreshDuration: defaultRefreshInterval,
	}
	if cfg.RefreshInterval != nil {
		dur := &durationpb.Duration{}
		if err := protojson.Unmarshal(cfg.RefreshInterval, dur); err != nil {
			return Options{}, fmt.Errorf("pemfile: protojson.Unmarshal(%+v) failed: %v", cfg.RefreshInterval, err)
		}
		opts.RefreshDuration = dur.AsDuration()
	}

	if err := opts.validate(); err != nil {
		return Options{}, err
	}
	return opts, nil
}
