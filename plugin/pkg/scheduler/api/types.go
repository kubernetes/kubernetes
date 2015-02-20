/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Where possible, json tags match the cli argument names.
// Top level config objects and all values required for proper functioning are not "omitempty".  Any truly optional piece of config is allowed to be omitted.

type Policy struct {
	api.TypeMeta `json:",inline"`
	Predicates   []PredicatePolicy `json:"predicates"`
	Priorities   []PriorityPolicy  `json:"priorities"`
}

type PredicatePolicy struct {
	Name     string             `json:"name"`
	Argument *PredicateArgument `json:"argument"`
}

type PriorityPolicy struct {
	Name     string            `json:"name"`
	Weight   int               `json:"weight"`
	Argument *PriorityArgument `json:"argument"`
}

// PredicateArgument represents the arguments that the different types of predicates take.
// Only one of its members may be specified.
type PredicateArgument struct {
	ServiceAffinity *ServiceAffinity `json:"serviceAffinity"`
	LabelsPresence  *LabelsPresence  `json:"labelsPresence"`
}

// PriorityArgument represents the arguments that the different types of priorities take.
// Only one of its members may be specified.
type PriorityArgument struct {
	ServiceAntiAffinity *ServiceAntiAffinity `json:"serviceAntiAffinity"`
	LabelPreference     *LabelPreference     `json:"labelPreference"`
}

type ServiceAffinity struct {
	Labels []string `json:"labels"`
}

type LabelsPresence struct {
	Labels   []string `json:"labels"`
	Presence bool     `json:"presence"`
}

type ServiceAntiAffinity struct {
	Label string `json:"label"`
}

type LabelPreference struct {
	Label    string `json:"label"`
	Presence bool   `json:"presence"`
}
