// +build !dockerless

/*
Copyright 2014 The Kubernetes Authors.

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

package network

// TODO: Consider making this value configurable.
const DefaultInterfaceName = "eth0"

// CNITimeoutSec is set to be slightly less than 240sec/4mins, which is the default remote runtime request timeout.
const CNITimeoutSec = 220

// UseDefaultMTU is a marker value that indicates the plugin should determine its own MTU
// It is the zero value, so a non-initialized value will mean "UseDefault"
const UseDefaultMTU = 0
