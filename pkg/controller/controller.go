/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package controller

// Controller represents a long-lived process that watches objects in the system and responds to events.
// All start/start complexity (managing stop channels, etc.) is deliberately left to the plugin.
type Controller interface {
	// Run starts a controller.  This function must be idempotent.
	Run() error
	// Stop stops a controller.  This function must be idempotent.
	Stop() error
	// Status returns runtime status of the controller and any potential error that may have stopped it
	Status() (bool, error)
}
