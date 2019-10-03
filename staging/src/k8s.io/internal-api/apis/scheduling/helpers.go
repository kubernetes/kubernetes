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

package scheduling

import (
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SystemPriorityClasses define system priority classes that are auto-created at cluster bootstrapping.
// Our API validation logic ensures that any priority class that has a system prefix or its value
// is higher than HighestUserDefinablePriority is equal to one of these SystemPriorityClasses.
var systemPriorityClasses = []*PriorityClass{
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: SystemNodeCritical,
		},
		Value:       SystemCriticalPriority + 1000,
		Description: "Used for system critical pods that must not be moved from their current node.",
	},
	{
		ObjectMeta: metav1.ObjectMeta{
			Name: SystemClusterCritical,
		},
		Value:       SystemCriticalPriority,
		Description: "Used for system critical pods that must run in the cluster, but can be moved to another node if necessary.",
	},
}

// SystemPriorityClasses returns the list of system priority classes.
// NOTE: be careful not to modify any of elements of the returned array directly.
func SystemPriorityClasses() []*PriorityClass {
	return systemPriorityClasses
}

// IsKnownSystemPriorityClass checks that "pc" is equal to one of the system PriorityClasses.
// It ignores "description", labels, annotations, etc. of the PriorityClass.
func IsKnownSystemPriorityClass(pc *PriorityClass) (bool, error) {
	for _, spc := range systemPriorityClasses {
		if spc.Name == pc.Name {
			if spc.Value != pc.Value {
				return false, fmt.Errorf("value of %v PriorityClass must be %v", spc.Name, spc.Value)
			}
			if spc.GlobalDefault != pc.GlobalDefault {
				return false, fmt.Errorf("globalDefault of %v PriorityClass must be %v", spc.Name, spc.GlobalDefault)
			}
			return true, nil
		}
	}
	return false, fmt.Errorf("%v is not a known system priority class", pc.Name)
}
