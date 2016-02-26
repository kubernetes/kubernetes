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

package rkt

import (
	"flag"
	"strings"
)

const DefaultOptions = "--dir=/var/lib/rkt --insecure-options=image,ondisk --local-config=/etc/rkt"

// Config stores the global configuration for the rkt runtime.
// Detailed documents can be found at:
// https://github.com/coreos/rkt/blob/master/Documentation/commands.md#global-options
type Config struct {
	// The absolute path to the binary, or leave empty to find it in $PATH.
	Path string
	// The debug flag for rkt.
	Debug bool
	// The rkt data directory.
	Dir string
	// Comma-separated list of security features to disable.
	// Allowed values: "none", "image", "tls", "ondisk", "http", "all".
	InsecureOptions string
	// Local configuration directory.
	LocalConfig string
	// System configuration directory.
	SystemConfig string
	// Automatically trust gpg keys fetched from https.
	TrustKeysFromHttps bool
	// User configuration directory.
	UserConfig string

	// The Option strings. This string passed to rkt directly.
	Options string
}

func NewConfig(path, options string) (*Config, error) {
	fs := flag.NewFlagSet("rktflags", flag.ContinueOnError)
	config := &Config{
		Path:    path,
		Options: options,
	}

	fs.BoolVar(&config.Debug, "debug", false, "print out more debug information to stderr")
	fs.StringVar(&config.Dir, "dir", "", "rkt data directory")
	fs.StringVar(&config.InsecureOptions, "insecure-options", "", `comma-separated list of security features to disable. Allowed values: "none", "image", "tls", "ondisk", "http", "all"`)
	fs.StringVar(&config.LocalConfig, "local-config", "", "local configuration directory")
	fs.StringVar(&config.SystemConfig, "system-config", "", "system configuration directory")
	fs.BoolVar(&config.TrustKeysFromHttps, "trust-keys-from-https", false, "automatically trust gpg keys fetched from https")
	fs.StringVar(&config.UserConfig, "user-config", "", "user configuration directory")

	if err := fs.Parse(strings.Fields(options)); err != nil {
		return nil, err
	}
	return config, nil
}

// buildGlobalOptions returns an array of global command line options.
func (c *Config) buildGlobalOptions() []string {
	return strings.Fields(c.Options)
}
