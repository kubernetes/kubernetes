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
	config              *internal.LeaderMigrationConfiguration
	migratedControllers map[string]bool
	// component indicates the name of the control-plane component that uses leader migration,
	//  which should be a controller manager, i.e. kube-controller-manager or cloud-controller-manager
	component string
}

// FilterFunc takes a name of controller, returning whether the controller should be started.
type FilterFunc func(controllerName string) bool

// NewLeaderMigrator creates a LeaderMigrator with given config for the given component. The component
//  indicates which controller manager is requesting this leader migration, and it should be consistent
//  with the component field of ControllerLeaderConfiguration.
func NewLeaderMigrator(config *internal.LeaderMigrationConfiguration, component string) *LeaderMigrator {
	migratedControllers := make(map[string]bool)
	for _, leader := range config.ControllerLeaders {
		migratedControllers[leader.Name] = leader.Component == component
	}
	return &LeaderMigrator{
		config:              config,
		migratedControllers: migratedControllers,
		component:           component,
	}
}

// FilterFunc returns the filter function that, when migrated == true
//  - returns true if the controller should start under the migration lock
//  - returns false if the controller should start under the main lock
// when migrated == false, the result is inverted.
func (m *LeaderMigrator) FilterFunc(migrated bool) FilterFunc {
	return func(controllerName string) bool {
		shouldRun, ok := m.migratedControllers[controllerName]
		if !ok {
			// The controller is not included in the migration
			// If the caller wants the controllers outside migration, then we should include it.
			return !migrated
		}
		// The controller is included in the migration
		// If the caller wants the controllers within migration, we should only include it
		//  if current component should run the controller
		return migrated && shouldRun
	}
}
