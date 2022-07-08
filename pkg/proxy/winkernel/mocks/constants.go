//go:build windows
// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package mocks

const (
	NwAdapterName          = "nw-adapter-123"
	NwType                 = "nw-type-123"
	TestHostName           = "test-hostname"
	TestNwName             = "TestNetwork"
	TestNwID               = "TestNetworkID"
	HnsID                  = "123ABC"
	IpAddress              = "192.168.1.2"
	MacAddress             = "00-11-22-33-44-55"
	ClusterCIDR            = "192.168.1.0/24"
	DestinationPrefix      = "192.168.2.0/24"
	ProviderAddress        = "10.0.0.1"
	Guid                   = "123ABC"
	SvcIP                  = "192.168.2.3"
	SvcPort                = 80
	SvcNodePort            = 3001
	SvcExternalIPs         = "50.60.70.81"
	SvcLBIP                = "11.21.31.41"
	SvcHealthCheckNodePort = 30000
	LbInternalPort         = 0x50
	LbExternalPort         = 0x50
	LbTCPProtocol          = 0x6
	HnsEndPointName        = "TestHnsEndpointName"
	EmptyID                = ""
)
