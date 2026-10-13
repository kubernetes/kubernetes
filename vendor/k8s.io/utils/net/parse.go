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

package net

import (
	forkednet "k8s.io/utils/internal/third_party/forked/golang/net"
)

// ParseIPSloppy is identical to Go's standard net.ParseIP, except that it allows
// leading '0' characters on numbers.  Go used to allow this and then changed
// the behavior in 1.17.  We're choosing to keep it for compat with potential
// stored values.
var ParseIPSloppy = forkednet.ParseIP

// ParseCIDRSloppy is identical to Go's standard net.ParseCIDR, except that it allows
// leading '0' characters on numbers.  Go used to allow this and then changed
// the behavior in 1.17.  We're choosing to keep it for compat with potential
// stored values.
var ParseCIDRSloppy = forkednet.ParseCIDR
