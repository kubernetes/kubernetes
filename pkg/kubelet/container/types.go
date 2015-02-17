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

package container

// Container represents a container.
type Container struct {
	ID         string   `json:"id"`
	Image      string   `json:"image,omitempty"`
	Command    string   `json:"command,omitempty"`
	Created    int64    `json:"created,omitempty"`
	Status     string   `json:"status,omitempty"`
	Ports      []Port   `json:"ports,omitempty"`
	SizeRw     int64    `json:"sizeRw,omitempty"`
	SizeRootFs int64    `json:"sizeRootFs,omitempty"`
	Names      []string `json:"names,omitempty"`
}

// Port is a type that represents a port mapping.
type Port struct {
	PrivatePort int64  `json:"PrivatePort,omitempty" yaml:"PrivatePort,omitempty"`
	PublicPort  int64  `json:"PublicPort,omitempty" yaml:"PublicPort,omitempty"`
	Type        string `json:"Type,omitempty" yaml:"Type,omitempty"`
	IP          string `json:"IP,omitempty" yaml:"IP,omitempty"`
}

// ListContainersOptions specify parameters to the ListContainers function.
type ListContainersOptions struct {
	All bool
}
