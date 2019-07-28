/*
Copyright 2016 The Kubernetes Authors.

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

package initsystem

// InitSystem is the interface that describe behaviors of an init system
type InitSystem interface {
	// return a string describing how to enable a service
	EnableCommand(service string) string

	// ServiceStart tries to start a specific service
	ServiceStart(service string) error

	// ServiceStop tries to stop a specific service
	ServiceStop(service string) error

	// ServiceRestart tries to reload the environment and restart the specific service
	ServiceRestart(service string) error

	// ServiceExists ensures the service is defined for this init system.
	ServiceExists(service string) bool

	// ServiceIsEnabled ensures the service is enabled to start on each boot.
	ServiceIsEnabled(service string) bool

	// ServiceIsActive ensures the service is running, or attempting to run. (crash looping in the case of kubelet)
	ServiceIsActive(service string) bool
}
