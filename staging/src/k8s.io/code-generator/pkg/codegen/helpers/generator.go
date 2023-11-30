/*
Copyright 2023 The Kubernetes Authors.

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

package helpers

import (
	goflag "flag"

	"k8s.io/klog/v2"
)

// Generator will generate the helpers for the given args.
type Generator struct {
	Flags *goflag.FlagSet
}

func (g *Generator) Generate(args *Args) error {
	klog.V(1).Infof("Generating helpers for %s", args.InputDir)
	klog.V(2).Infof("Helpers generator config %#v", args)

	if err := g.generateDeepCopy(args); err != nil {
		return err
	}

	if err := g.generateDefaulter(args); err != nil {
		return err
	}

	if err := g.generateConversion(args); err != nil {
		return err
	}

	klog.V(1).Infof("Successfully generated helpers for %v",
		args.InputDir)
	return nil
}
