/*
Copyright 2015 The Kubernetes Authors.

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

package hyperkube

import (
	"github.com/spf13/pflag"
)

var (
	nilKube = &nilKubeType{}
)

type Interface interface {
	// FindServer will find a specific server named name.
	FindServer(name string) bool

	// The executable name, used for help and soft-link invocation
	Name() string

	// Flags returns a flagset for "global" flags.
	Flags() *pflag.FlagSet
}

type nilKubeType struct{}

func (n *nilKubeType) FindServer(_ string) bool {
	return false
}

func (n *nilKubeType) Name() string {
	return ""
}

func (n *nilKubeType) Flags() *pflag.FlagSet {
	return nil
}

func Nil() Interface {
	return nilKube
}
