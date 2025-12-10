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

// FilterResult indicates whether and how the controller manager should start the controller.
type FilterResult int32

const (
	// ControllerUnowned indicates that the controller is owned by another controller manager
	//  and thus should NOT be started by this controller manager.
	ControllerUnowned = iota
	// ControllerMigrated indicates that the controller manager should start this controller
	//  with the migration lock.
	ControllerMigrated
	// ControllerNonMigrated indicates that the controller manager should start this controller
	//  with the main lock.
	ControllerNonMigrated
)

// FilterFunc takes a name of controller, returning a FilterResult indicating how to start controller.
type FilterFunc func(controllerName string) FilterResult
