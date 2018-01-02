// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flag

import "fmt"

// DiscardFlag is a flag type that discards all options with a warning
type DiscardFlag struct {
	Name string
	used bool // pflag calls .set at least once always
}

func NewDiscardFlag(name string) *DiscardFlag {
	df := DiscardFlag{
		Name: name,
	}
	return &df
}

func (df *DiscardFlag) Set(_ string) error {
	fmt.Printf("Warning: -%s not supported; ignoring\n", df.Name)
	return nil
}

func (df *DiscardFlag) String() string {
	return ""
}

func (df *DiscardFlag) Type() string {
	return "DiscardFlag"
}
