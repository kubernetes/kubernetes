/*
Copyright 2021 The Kubernetes Authors.

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

package leadermigration

import (
	internal "k8s.io/controller-manager/config"
)

// LeaderMigrator holds information required by the leader migration process.
type LeaderMigrator struct {
	// MigrationReady is closed after the controller manager finishes preparing for the migration lock.
	// After this point, the leader migration process will proceed to acquire the migration lock.
	MigrationReady chan struct{}

	// FilterFunc returns a FilterResult telling the controller manager what to do with the controller.
	FilterFunc FilterFunc
}

// NewLeaderMigrator creates a LeaderMigrator with given config for the given component. component
//
//	indicates which controller manager is requesting this leader migration, and it should be consistent
//	with the component field of ControllerLeaderConfiguration.
func NewLeaderMigrator(config *internal.LeaderMigrationConfiguration, component string) *LeaderMigrator {
	migratedControllers := make(map[string]bool)
	for _, leader := range config.ControllerLeaders {
		migratedControllers[leader.Name] = leader.Component == component || leader.Component == "*"
	}
	return &LeaderMigrator{
		MigrationReady: make(chan struct{}),
		FilterFunc: func(controllerName string) FilterResult {
			shouldRun, ok := migratedControllers[controllerName]
			if ok {
				// The controller is included in the migration
				if shouldRun {
					// If the controller manager should run the controller,
					//  start it in the migration lock.
					return ControllerMigrated
				}
				// Otherwise, the controller should be started by
				//  some other controller manager.
				return ControllerUnowned
			}
			// The controller is not included in the migration,
			//  and should be started in the main lock.
			return ControllerNonMigrated
		},
	}
}
