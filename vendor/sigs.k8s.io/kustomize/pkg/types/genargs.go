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

package types

// GenArgs contains both generator args and options
type GenArgs struct {
	args *GeneratorArgs
	opts *GeneratorOptions
}

// NewGenArgs returns a new object of GenArgs
func NewGenArgs(args *GeneratorArgs, opts *GeneratorOptions) *GenArgs {
	return &GenArgs{
		args: args,
		opts: opts,
	}
}

// NeedHashSuffix returns true if the hash suffix is needed.
// It is needed when the two conditions are both met
//  1) GenArgs is not nil
//  2) DisableNameSuffixHash in GeneratorOptions is not set to true
func (g *GenArgs) NeedsHashSuffix() bool {
	return g.args != nil && (g.opts == nil || g.opts.DisableNameSuffixHash == false)
}

// Behavior returns Behavior field of GeneratorArgs
func (g *GenArgs) Behavior() GenerationBehavior {
	if g.args == nil {
		return BehaviorUnspecified
	}
	return NewGenerationBehavior(g.args.Behavior)
}
