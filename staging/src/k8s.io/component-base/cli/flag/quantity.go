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

package flag

import (
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/api/resource"
)

// qFlag is a helper type for the Flag function
type qFlag struct {
	dest *resource.Quantity
}

// Sets the value of the internal Quantity. (used by flag & pflag)
func (qf qFlag) Set(val string) error {
	q, err := resource.ParseQuantity(val)
	if err != nil {
		return err
	}
	// This copy is OK because q will not be referenced again.
	*qf.dest = q
	return nil
}

// Converts the value of the internal Quantity to a string. (used by flag & pflag)
func (qf qFlag) String() string {
	return qf.dest.String()
}

// States the type of flag this is (Quantity). (used by pflag)
func (qf qFlag) Type() string {
	return "quantity"
}

// QuantityFlag is a helper that makes a quantity flag (using the pflag
// package). Will panic if defaultValue is not a valid quantity. To use a
// FlagSet, see NewQuantityFlagValue.
func QuantityFlag(flagName, defaultValue, description string) *resource.Quantity {
	q := resource.MustParse(defaultValue)
	pflag.Var(NewQuantityFlagValue(&q), flagName, description)
	return &q
}

// NewQuantityFlagValue returns an object that can be used to back a flag,
// pointing at the given Quantity variable. This can be used with pflag.Var.
func NewQuantityFlagValue(q *resource.Quantity) pflag.Value {
	return qFlag{q}
}
