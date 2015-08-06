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

package abac

// Policy authorizes Kubernetes API actions using an Attribute-based access
// control scheme.

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"

	"k8s.io/kubernetes/pkg/auth/authorizer"
)

// TODO: make this into a real API object.  Note that when that happens, it
// will get MetaData.  However, the Kind and Namespace in the struct below
// will be separate from the Kind and Namespace in the Metadata.  Obviously,
// meta.Kind will be something like policy, and policy.Kind has to be allowed
// to be different.  Less obviously, namespace needs to be different as well.
// This will allow wildcard matching strings to be used in the future for the
// body.Namespace, if we want to add that feature, without affecting the
// meta.Namespace.
type policy struct {
	User  string `json:"user,omitempty"`
	Group string `json:"group,omitempty"`
	// TODO: add support for robot accounts as well as human user accounts.
	// TODO: decide how to namespace user names when multiple authentication
	// providers are in use. Either add "Realm", or assume "user@example.com"
	// format.

	// TODO: Make the "cluster" Kinds be one API group (minions, bindings,
	// events, endpoints).  The "user" Kinds are another (pods, services,
	// replicationControllers, operations) Make a "plugin", e.g. build
	// controller, be another group.  That way when we add a new object to a
	// the API, we don't have to add lots of policy?

	// TODO: make this a proper REST object with its own registry.
	Readonly  bool   `json:"readonly,omitempty"`
	Resource  string `json:"resource,omitempty"`
	Namespace string `json:"namespace,omitempty"`

	// TODO: "expires" string in RFC3339 format.

	// TODO: want a way to allow some users to restart containers of a pod but
	// not delete or modify it.

	// TODO: want a way to allow a controller to create a pod based only on a
	// certain podTemplates.
}

type policyList []policy

// TODO: Have policies be created via an API call and stored in REST storage.
func NewFromFile(path string) (policyList, error) {
	// File format is one map per line.  This allows easy concatentation of files,
	// comments in files, and identification of errors by line number.
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	pl := make(policyList, 0)

	for scanner.Scan() {
		var p policy
		b := scanner.Bytes()
		// TODO: skip comment lines.
		err = json.Unmarshal(b, &p)
		if err != nil {
			// TODO: line number in errors.
			return nil, err
		}
		pl = append(pl, p)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return pl, nil
}

func (p policy) matches(a authorizer.Attributes) bool {
	if p.subjectMatches(a) {
		if p.Readonly == false || (p.Readonly == a.IsReadOnly()) {
			if p.Resource == "" || (p.Resource == a.GetResource()) {
				if p.Namespace == "" || (p.Namespace == a.GetNamespace()) {
					return true
				}
			}
		}
	}
	return false
}

func (p policy) subjectMatches(a authorizer.Attributes) bool {
	if p.User != "" {
		// Require user match
		if p.User != a.GetUserName() {
			return false
		}
	}

	if p.Group != "" {
		// Require group match
		for _, group := range a.GetGroups() {
			if p.Group == group {
				return true
			}
		}
		return false
	}

	return true
}

// Authorizer implements authorizer.Authorize
func (pl policyList) Authorize(a authorizer.Attributes) error {
	for _, p := range pl {
		if p.matches(a) {
			return nil
		}
	}
	return errors.New("No policy matched.")
	// TODO: Benchmark how much time policy matching takes with a medium size
	// policy file, compared to other steps such as encoding/decoding.
	// Then, add Caching only if needed.
}
