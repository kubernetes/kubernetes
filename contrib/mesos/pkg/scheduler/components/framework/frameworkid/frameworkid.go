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

package frameworkid

import (
	"errors"

	"golang.org/x/net/context"
)

type (
	// LookupFunc retrieves a framework ID from persistent storage
	LookupFunc func(context.Context) (string, error)

	// StoreFunc stores a framework ID in persistent storage
	StoreFunc func(context.Context, string) error

	// RemoveFunc removes a framework ID from persistent storage
	RemoveFunc func(context.Context) error

	Getter interface {
		Get(context.Context) (string, error)
	}

	Setter interface {
		Set(context.Context, string) error
	}

	Remover interface {
		Remove(context.Context) error
	}

	Storage interface {
		Getter
		Setter
		Remover
	}
)

var ErrMismatch = errors.New("framework ID mismatch")

func (f LookupFunc) Get(c context.Context) (string, error) { return f(c) }
func (f StoreFunc) Set(c context.Context, id string) error { return f(c, id) }
func (f RemoveFunc) Remove(c context.Context) error        { return f(c) }
