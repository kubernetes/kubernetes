// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build host coreos src

package main

import "testing"

func TestNetHost(t *testing.T) {
	NewNetHostTest().Execute(t)
}

func TestNetHostConnectivity(t *testing.T) {
	NewNetHostConnectivityTest().Execute(t)
}

func TestNetPortFwdConnectivity(t *testing.T) {
	NewNetPortFwdConnectivityTest(
		defaultSamePortFwdCase,
		defaultDiffPortFwdCase,
		defaultSpecificIPFwdCase,
		defaultSpecificIPFwdFailCase,
		defaultLoSamePortFwdCase,
		defaultLoDiffPortFwdCase,
		bridgeSamePortFwdCase,
		bridgeDiffPortFwdCase,
		bridgeLoSamePortFwdCase,
		bridgeLoDiffPortFwdCase,
	).Execute(t)
}

func TestNetNone(t *testing.T) {
	NewNetNoneTest().Execute(t)
}

func TestNetCustomMacvlan(t *testing.T) {
	NewNetCustomMacvlanTest().Execute(t)
}

func TestNetCustomBridge(t *testing.T) {
	NewNetCustomBridgeTest().Execute(t)
}

func TestNetOverride(t *testing.T) {
	NewNetOverrideTest().Execute(t)
}

func TestNetDefaultIPArg(t *testing.T) {
	NewNetDefaultIPArgTest().Execute(t)
}

func TestNetIPConflict(t *testing.T) {
	NewNetIPConflictTest().Execute(t)
}

func TestNetCustomPtp(t *testing.T) {
	// PTP means connection Point-To-Point. That is, connections to other pods/containers should be forbidden
	NewNetCustomPtpTest(true).Execute(t)
}

func TestNetDefaultConnectivity(t *testing.T) {
	NewNetDefaultConnectivityTest().Execute(t)
}

func TestNetDefaultNetNS(t *testing.T) {
	NewTestNetDefaultNetNS().Execute(t)
}

func TestNetLongName(t *testing.T) {
	NewTestNetLongName().Execute(t)
}

func TestNetDefaultRestrictedConnectivity(t *testing.T) {
	NewTestNetDefaultRestrictedConnectivity().Execute(t)
}

func TestNetDefaultGW(t *testing.T) {
	NewNetDefaultGWTest().Execute(t)
}

func TestNetCNIEnv(t *testing.T) {
	NewNetCNIEnvTest().Execute(t)
}

func TestNetCNIDNS(t *testing.T) {
	NewNetCNIDNSTest().Execute(t)
}

func TestNetCNIDNSArg(t *testing.T) {
	NewNetCNIDNSArgTest().Execute(t)
}

func TestNetCNIDNSArgNone(t *testing.T) {
	NewNetCNIDNSArgNoneTest().Execute(t)
}
