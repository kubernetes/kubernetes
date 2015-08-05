/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/cmd/km/server.go
package main

import (
	"io/ioutil"
	"strings"

	"k8s.io/kubernetes/pkg/util"

	"github.com/spf13/pflag"
)

type serverRunFunc func(s *Server, args []string) error

// Server describes a server that this binary can morph into.
type Server struct {
	SimpleUsage string        // One line description of the server.
	Long        string        // Longer free form description of the server
	Run         serverRunFunc // Run the server.  This is not expected to return.

	flags *pflag.FlagSet // Flags for the command (and all dependents)
	name  string
	hk    *HyperKube
}

// Usage returns the full usage string including all of the flags.
func (s *Server) Usage() error {
	tt := `{{if .Long}}{{.Long | trim | wrap ""}}
{{end}}Usage:
  {{.SimpleUsage}} [flags]

Available Flags:
{{.Flags.FlagUsages}}`

	return util.ExecuteTemplate(s.hk.Out(), tt, s)
}

// Name returns the name of the command as derived from the usage line.
func (s *Server) Name() string {
	if s.name != "" {
		return s.name
	}
	name := s.SimpleUsage
	i := strings.Index(name, " ")
	if i >= 0 {
		name = name[:i]
	}
	return name
}

// Flags returns a flagset for this server
func (s *Server) Flags() *pflag.FlagSet {
	if s.flags == nil {
		s.flags = pflag.NewFlagSet(s.Name(), pflag.ContinueOnError)
		s.flags.SetOutput(ioutil.Discard)
		s.flags.SetNormalizeFunc(util.WordSepNormalizeFunc)
	}
	return s.flags
}
