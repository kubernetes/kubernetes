/*
Copyright 2018 The Kubernetes Authors.

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

package componentconfigs

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
)

// UnsupportedConfigVersionError is a special error type returned whenever we encounter too old config version
type UnsupportedConfigVersionError struct {
	// OldVersion is the config version that is causing the problem
	OldVersion schema.GroupVersion

	// CurrentVersion describes the natively supported config version
	CurrentVersion schema.GroupVersion

	// Document points to the YAML/JSON document that caused the problem
	Document []byte
}

// Error implements the standard Golang error interface for UnsupportedConfigVersionError
func (err *UnsupportedConfigVersionError) Error() string {
	return fmt.Sprintf("unsupported apiVersion %q, you may have to do manual conversion to %q and run kubeadm again", err.OldVersion, err.CurrentVersion)
}

// UnsupportedConfigVersionsErrorMap is a cumulative version of the UnsupportedConfigVersionError type
type UnsupportedConfigVersionsErrorMap map[string]*UnsupportedConfigVersionError

// Error implements the standard Golang error interface for UnsupportedConfigVersionsErrorMap
func (errs UnsupportedConfigVersionsErrorMap) Error() string {
	// Make sure the error messages we print are predictable by sorting them by the group names involved
	groups := make([]string, 0, len(errs))
	for group := range errs {
		groups = append(groups, group)
	}
	sort.Strings(groups)

	msgs := make([]string, 1, 1+len(errs))
	msgs[0] = "multiple unsupported config version errors encountered:"
	for _, group := range groups {
		msgs = append(msgs, errs[group].Error())
	}

	return strings.Join(msgs, "\n\t- ")
}

// warnDefaultComponentConfigValue prints a warning if the user modified a field in a certain
// ComponentConfig from the default recommended value in kubeadm.
func warnDefaultComponentConfigValue(componentConfigKind, paramName string, defaultValue, userValue interface{}) {
	klog.Warningf("The recommended value for %q in %q is: %v; the provided value is: %v",
		paramName, componentConfigKind, defaultValue, userValue)
}
