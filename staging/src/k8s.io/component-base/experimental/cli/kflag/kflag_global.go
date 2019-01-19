/*
Copyright 2019 The Kubernetes Authors.

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

package kflag

import (
	"flag"
	"fmt"
	"strings"

	"github.com/spf13/pflag"
)

// Based on the Kubelet's k8s.io/kubernetes/cmd/kubelet/app/options/globalflags.go

func (fs *FlagSet) MustAddGlobalFlag(name string) {
	if f := flag.CommandLine.Lookup(name); f != nil {
		pflagFlag := pflag.PFlagFromGoFlag(f)
		pflagFlag.Name = normalize(pflagFlag.Name)
		fs.fs.AddFlag(pflagFlag)
	} else {
		panic(fmt.Sprintf("failed to find flag in global flagset (flag): %s", name))
	}
}

func (fs *FlagSet) MustAddGlobalPflag(name string) {
	if f := pflag.CommandLine.Lookup(name); f != nil {
		f.Name = normalize(f.Name)
		fs.fs.AddFlag(f)
	} else {
		panic(fmt.Sprintf("failed to find flag in global flagset (pflag): %s", name))
	}
}

// For legacy reasons, components may wish to mark global
// flags deprecated before removing them:

const deprecated = "This flag has been deprecated and will be removed in a future version."

func (fs *FlagSet) MustAddDeprecatedGlobalFlag(name string) {
	fs.MustAddGlobalFlag(name)
	fs.fs.Lookup(normalize(name)).Deprecated = deprecated
}

func (fs *FlagSet) MustAddDeprecatedGlobalPflag(name string) {
	fs.MustAddGlobalPflag(name)
	fs.fs.Lookup(normalize(name)).Deprecated = deprecated
}

// normalize replaces underscores with hyphens
func normalize(s string) string {
	return strings.Replace(s, "_", "-", -1)
}
