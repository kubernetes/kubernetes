/*
Copyright 2014 The Kubernetes Authors.

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

package util

import (
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// resourceForms holds resource long and short form (name and alias).
type resourceForms struct {
	ShortForm schema.GroupResource
	LongForm  schema.GroupResource
}

//
// TODO(tcharding): Move validResources string out of `kubectl/cmd.go` to here.
// This will mean there is a single point of knowledge (SPOK).
//

// resourceTab is a static list of resources that have short forms (aliases).
var resourceTab = []resourceForms{
	// Note that the list is ordered by group.
	{
		ShortForm: schema.GroupResource{Resource: "cm"},
		LongForm:  schema.GroupResource{Resource: "configmaps"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "cs"},
		LongForm:  schema.GroupResource{Resource: "componentstatuses"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "ep"},
		LongForm:  schema.GroupResource{Resource: "endpoints"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "ev"},
		LongForm:  schema.GroupResource{Resource: "events"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "limits"},
		LongForm:  schema.GroupResource{Resource: "limitranges"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "no"},
		LongForm:  schema.GroupResource{Resource: "nodes"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "ns"},
		LongForm:  schema.GroupResource{Resource: "namespaces"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "po"},
		LongForm:  schema.GroupResource{Resource: "pods"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "pvc"},
		LongForm:  schema.GroupResource{Resource: "persistentvolumeclaims"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "pv"},
		LongForm:  schema.GroupResource{Resource: "persistentvolumes"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "quota"},
		LongForm:  schema.GroupResource{Resource: "resourcequotas"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "rc"},
		LongForm:  schema.GroupResource{Resource: "replicationcontrollers"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "rs"},
		LongForm:  schema.GroupResource{Resource: "replicasets"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "sa"},
		LongForm:  schema.GroupResource{Resource: "serviceaccounts"},
	},
	{
		ShortForm: schema.GroupResource{Resource: "svc"},
		LongForm:  schema.GroupResource{Resource: "services"},
	},
	{
		ShortForm: schema.GroupResource{Group: "autoscaling", Resource: "hpa"},
		LongForm:  schema.GroupResource{Group: "autoscaling", Resource: "horizontalpodautoscalers"},
	},
	{
		ShortForm: schema.GroupResource{Group: "certificates.k8s.io", Resource: "csr"},
		LongForm:  schema.GroupResource{Group: "certificates.k8s.io", Resource: "certificatesigningrequests"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "deploy"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "deployments"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "ds"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "daemonsets"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "hpa"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "horizontalpodautoscalers"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "ing"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "ingresses"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "netpol"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "networkpolicies"},
	},
	{
		ShortForm: schema.GroupResource{Group: "extensions", Resource: "psp"},
		LongForm:  schema.GroupResource{Group: "extensions", Resource: "podSecurityPolicies"},
	},
	{
		ShortForm: schema.GroupResource{Group: "policy", Resource: "pdb"},
		LongForm:  schema.GroupResource{Group: "policy", Resource: "poddisruptionbudgets"},
	},
}

// ResourceShortFormFor looks for a short form of resource given name.
// TODO: Change the signature of this function so that it can
// make use of ResourceShortcuts.
func ResourceShortFormFor(longForm string) (string, bool) {

	// Search resource table for matching longForm.
	for _, r := range resourceTab {
		if r.LongForm.Resource == longForm {
			return r.ShortForm.Resource, true
		}
	}
	return "", false
}

// ResourceAliases returns the resource shortcuts and plural forms for the given resources.
func ResourceAliases(rs []string) []string {
	as := make([]string, 0, len(rs))
	plurals := make(map[string]struct{}, len(rs))
	for _, r := range rs {
		var plural string
		switch {
		case r == "endpoints":
			plural = r // exception. "endpoint" does not exist. Why?
		case strings.HasSuffix(r, "y"):
			plural = r[0:len(r)-1] + "ies"
		case strings.HasSuffix(r, "s"):
			plural = r + "es"
		default:
			plural = r + "s"
		}
		as = append(as, plural)

		plurals[plural] = struct{}{}
	}

	for _, item := range resourceTab {
		if _, found := plurals[item.LongForm.Resource]; found {
			as = append(as, item.ShortForm.Resource)
		}
	}
	return as
}
