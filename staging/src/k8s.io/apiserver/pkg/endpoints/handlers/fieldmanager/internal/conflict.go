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

package internal

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
)

// NewConflictError returns an error including details on the requests apply conflicts
func NewConflictError(conflicts merge.Conflicts) *errors.StatusError {
	causes := []metav1.StatusCause{}
	for _, conflict := range conflicts {
		causes = append(causes, metav1.StatusCause{
			Type:    metav1.CauseTypeFieldManagerConflict,
			Message: fmt.Sprintf("conflict with %v", printManager(conflict.Manager)),
			Field:   conflict.Path.String(),
		})
	}
	return errors.NewApplyConflict(causes, getConflictMessage(conflicts))
}

func getConflictMessage(conflicts merge.Conflicts) string {
	if len(conflicts) == 1 {
		return fmt.Sprintf("Apply failed with 1 conflict: conflict with %v: %v", printManager(conflicts[0].Manager), conflicts[0].Path)
	}

	m := map[string][]fieldpath.Path{}
	for _, conflict := range conflicts {
		m[conflict.Manager] = append(m[conflict.Manager], conflict.Path)
	}

	uniqueManagers := []string{}
	for manager := range m {
		uniqueManagers = append(uniqueManagers, manager)
	}

	// Print conflicts by sorted managers.
	sort.Strings(uniqueManagers)

	messages := []string{}
	for _, manager := range uniqueManagers {
		messages = append(messages, fmt.Sprintf("conflicts with %v:", printManager(manager)))
		for _, path := range m[manager] {
			messages = append(messages, fmt.Sprintf("- %v", path))
		}
	}
	return fmt.Sprintf("Apply failed with %d conflicts: %s", len(conflicts), strings.Join(messages, "\n"))
}

func printManager(manager string) string {
	encodedManager := &metav1.ManagedFieldsEntry{}
	if err := json.Unmarshal([]byte(manager), encodedManager); err != nil {
		return fmt.Sprintf("%q", manager)
	}
	if encodedManager.Operation == metav1.ManagedFieldsOperationUpdate {
		if encodedManager.Time == nil {
			return fmt.Sprintf("%q using %v", encodedManager.Manager, encodedManager.APIVersion)
		}
		return fmt.Sprintf("%q using %v at %v", encodedManager.Manager, encodedManager.APIVersion, encodedManager.Time.UTC().Format(time.RFC3339))
	}
	return fmt.Sprintf("%q", encodedManager.Manager)
}
