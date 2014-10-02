/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

// FlagSet abstracts the flag interface for compatibility with both Golang "flag"
// and cobra pflags (Posix style).
type FlagSet interface {
	StringVar(p *string, name, value, usage string)
	BoolVar(p *bool, name string, value bool, usage string)
}

// BindClientConfigFlags registers a standard set of CLI flags for connecting to a Kubernetes API server.
func BindClientConfigFlags(flags FlagSet, config *Config) {
	flags.StringVar(&config.Host, "master", config.Host, "The address of the Kubernetes API server")
	flags.StringVar(&config.Version, "api_version", config.Version, "The API version to use when talking to the server")
	flags.BoolVar(&config.Insecure, "insecure_skip_tls_verify", config.Insecure, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.")
}
