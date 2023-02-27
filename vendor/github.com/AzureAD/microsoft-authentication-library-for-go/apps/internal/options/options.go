// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package options

import (
	"errors"
	"fmt"
)

// CallOption implements an optional argument to a method call. See
// https://blog.devgenius.io/go-call-option-that-can-be-used-with-multiple-methods-6c81734f3dbe
// for an explanation of the usage pattern.
type CallOption interface {
	Do(any) error
	callOption()
}

// ApplyOptions applies all the callOptions to options. options must be a pointer to a struct and
// callOptions must be a list of objects that implement CallOption.
func ApplyOptions[O, C any](options O, callOptions []C) error {
	for _, o := range callOptions {
		if t, ok := any(o).(CallOption); !ok {
			return fmt.Errorf("unexpected option type %T", o)
		} else if err := t.Do(options); err != nil {
			return err
		}
	}
	return nil
}

// NewCallOption returns a new CallOption whose Do() method calls function "f".
func NewCallOption(f func(any) error) CallOption {
	if f == nil {
		// This isn't a practical concern because only an MSAL maintainer can get
		// us here, by implementing a do-nothing option. But if someone does that,
		// the below ensures the method invoked with the option returns an error.
		return callOption(func(any) error {
			return errors.New("invalid option: missing implementation")
		})
	}
	return callOption(f)
}

// callOption is an adapter for a function to a CallOption
type callOption func(any) error

func (c callOption) Do(a any) error {
	return c(a)
}

func (callOption) callOption() {}
