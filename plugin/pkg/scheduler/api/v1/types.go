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

package v1

import (
	apiv1 "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
)

type Policy struct {
	apiv1.TypeMeta `json:",inline"`
	// Holds the information to configure the fit predicate functions
	Predicates []PredicatePolicy `json:"predicates"`
	// Holds the information to configure the priority functions
	Priorities []PriorityPolicy `json:"priorities"`
}

type PredicatePolicy struct {
	// Identifier of the predicate policy
	// For a custom predicate, the name can be user-defined
	// For the Kubernetes provided predicates, the name is the identifier of the pre-defined predicate
	Name string `json:"name"`
	// Holds the parameters to configure the given predicate
	Argument *PredicateArgument `json:"argument"`
}

type PriorityPolicy struct {
	// Identifier of the priority policy
	// For a custom priority, the name can be user-defined
	// For the Kubernetes provided priority functions, the name is the identifier of the pre-defined priority function
	Name string `json:"name"`
	// The numeric multiplier for the minion scores that the priority function generates
	// The weight should be non-zero and can be a positive or a negative integer
	Weight int `json:"weight"`
	// Holds the parameters to configure the given priority function
	Argument *PriorityArgument `json:"argument"`
}

// Represents the arguments that the different types of predicates take
// Only one of its members may be specified
type PredicateArgument struct {
	// The predicate that provides affinity for pods belonging to a service
	// It uses a label to identify minions that belong to the same "group"
	ServiceAffinity *ServiceAffinity `json:"serviceAffinity"`
	// The predicate that checks whether a particular minion has a certain label
	// defined or not, regardless of value
	LabelsPresence *LabelsPresence `json:"labelsPresence"`
}

// Represents the arguments that the different types of priorities take.
// Only one of its members may be specified
type PriorityArgument struct {
	// The priority function that ensures a good spread (anti-affinity) for pods belonging to a service
	// It uses a label to identify minions that belong to the same "group"
	ServiceAntiAffinity *ServiceAntiAffinity `json:"serviceAntiAffinity"`
	// The priority function that checks whether a particular minion has a certain label
	// defined or not, regardless of value
	LabelPreference *LabelPreference `json:"labelPreference"`
}

// Holds the parameters that are used to configure the corresponding predicate
type ServiceAffinity struct {
	// The list of labels that identify minion "groups"
	// All of the labels should match for the minion to be considered a fit for hosting the pod
	Labels []string `json:"labels"`
}

// Holds the parameters that are used to configure the corresponding predicate
type LabelsPresence struct {
	// The list of labels that identify minion "groups"
	// All of the labels should be either present (or absent) for the minion to be considered a fit for hosting the pod
	Labels []string `json:"labels"`
	// The boolean flag that indicates whether the labels should be present or absent from the minion
	Presence bool `json:"presence"`
}

// Holds the parameters that are used to configure the corresponding priority function
type ServiceAntiAffinity struct {
	// Used to identify minion "groups"
	Label string `json:"label"`
}

// Holds the parameters that are used to configure the corresponding priority function
type LabelPreference struct {
	// Used to identify minion "groups"
	Label string `json:"label"`
	// This is a boolean flag
	// If true, higher priority is given to minions that have the label
	// If false, higher priority is given to minions that do not have the label
	Presence bool `json:"presence"`
}
