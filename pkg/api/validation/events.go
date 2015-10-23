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

package validation

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// ValidateEvent makes sure that the event makes sense.
func ValidateEvent(event *api.Event) field.ErrorList {
	allErrs := field.ErrorList{}
	// There is no namespace required for root-scoped kind, for example, node.
	// However, older client code accidentally sets event.Namespace
	// to api.NamespaceDefault, so we accept that too, but "" is preferred.
	// Todo: Events may reference 3rd party object, and we can't check whether the object is namespaced.
	// Suppose them are namespaced. Do check if we can get the piece of information.
	// This should apply to all groups served by this apiserver.
	group := apiutil.GetGroup(event.InvolvedObject.APIVersion)
	version := apiutil.GetVersion(event.InvolvedObject.APIVersion)
	if (group == "" || group == "extensions") && (version != "extensions" && version != "") {
		namespacedKindFlag, err := isNamespacedKind(event.InvolvedObject.Kind, event.InvolvedObject.APIVersion)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "kind"), event.InvolvedObject.Kind, fmt.Sprintf("couldn't check whether namespace is allowed: %s", err)))
		} else {
			if !namespacedKindFlag &&
				event.Namespace != api.NamespaceDefault &&
				event.Namespace != "" {
				allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, fmt.Sprintf("not allowed for %s", event.InvolvedObject.Kind)))
			}
			if namespacedKindFlag &&
				event.Namespace != event.InvolvedObject.Namespace {
				allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match involvedObject"))
			}
		}
	} else {
		if event.Namespace != event.InvolvedObject.Namespace {
			allErrs = append(allErrs, field.Invalid(field.NewPath("involvedObject", "namespace"), event.InvolvedObject.Namespace, "does not match involvedObject"))
		}
	}

	if !validation.IsDNS1123Subdomain(event.Namespace) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("namespace"), event.Namespace, ""))
	}
	return allErrs
}

// Check whether the kind in groupVersion is scoped at the root of the api hierarchy
func isNamespacedKind(kind, groupVersion string) (bool, error) {
	group := apiutil.GetGroup(groupVersion)
	g, err := registered.Group(group)
	if err != nil {
		return false, err
	}
	restMapping, err := g.RESTMapper.RESTMapping(unversioned.GroupKind{Group: group, Kind: kind}, apiutil.GetVersion(groupVersion))
	if err != nil {
		return false, err
	}
	scopeName := restMapping.Scope.Name()
	if scopeName == meta.RESTScopeNameNamespace {
		return true, nil
	}
	return false, nil
}
