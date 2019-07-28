/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package esx

import "github.com/vmware/govmomi/vim25/types"

// PerfCounter is the default template for the PerformanceManager perfCounter property.
// Capture method:
//   govc object.collect -s -dump PerformanceManager:ha-perfmgr perfCounter
var PerfCounter = []types.PerfCounterInfo{
	{
		Key: 0,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "CPU usage as a percentage during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{1, 2, 3},
	},
	{
		Key: 1,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "CPU usage as a percentage during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 2,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "CPU usage as a percentage during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 3,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "CPU usage as a percentage during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 4,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage in MHz",
				Summary: "CPU usage in megahertz during the interval",
			},
			Key: "usagemhz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{5, 6, 7},
	},
	{
		Key: 5,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage in MHz",
				Summary: "CPU usage in megahertz during the interval",
			},
			Key: "usagemhz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 6,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage in MHz",
				Summary: "CPU usage in megahertz during the interval",
			},
			Key: "usagemhz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 7,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage in MHz",
				Summary: "CPU usage in megahertz during the interval",
			},
			Key: "usagemhz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 8,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reserved capacity",
				Summary: "Total CPU capacity reserved by virtual machines",
			},
			Key: "reservedCapacity",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 9,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "Amount of time spent on system processes on each virtual CPU in the virtual machine",
			},
			Key: "system",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 10,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Wait",
				Summary: "Total CPU time spent in wait state",
			},
			Key: "wait",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 11,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Ready",
				Summary: "Time that the virtual machine was ready, but could not get scheduled to run on the physical CPU during last measurement interval",
			},
			Key: "ready",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 12,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Used",
				Summary: "Total CPU usage",
			},
			Key: "used",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 13,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Idle",
				Summary: "Total time that the CPU spent in an idle state",
			},
			Key: "idle",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 14,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap wait",
				Summary: "CPU time spent waiting for swap-in",
			},
			Key: "swapwait",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 15,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "CPU utilization as a percentage during the interval (CPU usage and CPU utilization might be different due to power management technologies or hyper-threading)",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{16, 17, 18},
	},
	{
		Key: 16,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "CPU utilization as a percentage during the interval (CPU usage and CPU utilization might be different due to power management technologies or hyper-threading)",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 17,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "CPU utilization as a percentage during the interval (CPU usage and CPU utilization might be different due to power management technologies or hyper-threading)",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 18,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "CPU utilization as a percentage during the interval (CPU usage and CPU utilization might be different due to power management technologies or hyper-threading)",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 19,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Core Utilization",
				Summary: "CPU utilization of the corresponding core (if hyper-threading is enabled) as a percentage during the interval (A core is utilized if either or both of its logical CPUs are utilized)",
			},
			Key: "coreUtilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{20, 21, 22},
	},
	{
		Key: 20,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Core Utilization",
				Summary: "CPU utilization of the corresponding core (if hyper-threading is enabled) as a percentage during the interval (A core is utilized if either or both of its logical CPUs are utilized)",
			},
			Key: "coreUtilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 21,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Core Utilization",
				Summary: "CPU utilization of the corresponding core (if hyper-threading is enabled) as a percentage during the interval (A core is utilized if either or both of its logical CPUs are utilized)",
			},
			Key: "coreUtilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 22,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Core Utilization",
				Summary: "CPU utilization of the corresponding core (if hyper-threading is enabled) as a percentage during the interval (A core is utilized if either or both of its logical CPUs are utilized)",
			},
			Key: "coreUtilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 23,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Total capacity",
				Summary: "Total CPU capacity reserved by and available for virtual machines",
			},
			Key: "totalCapacity",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 24,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Latency",
				Summary: "Percent of time the virtual machine is unable to run because it is contending for access to the physical CPU(s)",
			},
			Key: "latency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 25,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Entitlement",
				Summary: "CPU resources devoted by the ESX scheduler",
			},
			Key: "entitlement",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 26,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Demand",
				Summary: "The amount of CPU resources a virtual machine would use if there were no CPU contention or CPU limit",
			},
			Key: "demand",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 27,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Co-stop",
				Summary: "Time the virtual machine is ready to run, but is unable to run due to co-scheduling constraints",
			},
			Key: "costop",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 28,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Max limited",
				Summary: "Time the virtual machine is ready to run, but is not run due to maxing out its CPU limit setting",
			},
			Key: "maxlimited",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 29,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overlap",
				Summary: "Time the virtual machine was interrupted to perform system services on behalf of itself or other virtual machines",
			},
			Key: "overlap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 30,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Run",
				Summary: "Time the virtual machine is scheduled to run",
			},
			Key: "run",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 31,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Demand-to-entitlement ratio",
				Summary: "CPU resource entitlement to CPU demand ratio (in percents)",
			},
			Key: "demandEntitlementRatio",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 32,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Readiness",
				Summary: "Percentage of time that the virtual machine was ready, but could not get scheduled to run on the physical CPU",
			},
			Key: "readiness",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU",
				Summary: "CPU",
			},
			Key: "cpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65536,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host consumed %",
				Summary: "Percentage of host physical memory that has been consumed",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65537, 65538, 65539},
	},
	{
		Key: 65537,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host consumed %",
				Summary: "Percentage of host physical memory that has been consumed",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65538,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host consumed %",
				Summary: "Percentage of host physical memory that has been consumed",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65539,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host consumed %",
				Summary: "Percentage of host physical memory that has been consumed",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65540,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Granted",
				Summary: "Amount of host physical memory or physical memory that is mapped for a virtual machine or a host",
			},
			Key: "granted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65541, 65542, 65543},
	},
	{
		Key: 65541,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Granted",
				Summary: "Amount of host physical memory or physical memory that is mapped for a virtual machine or a host",
			},
			Key: "granted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65542,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Granted",
				Summary: "Amount of host physical memory or physical memory that is mapped for a virtual machine or a host",
			},
			Key: "granted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65543,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Granted",
				Summary: "Amount of host physical memory or physical memory that is mapped for a virtual machine or a host",
			},
			Key: "granted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65544,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active",
				Summary: "Amount of guest physical memory that is being actively read or written by guest. Activeness is estimated by ESXi",
			},
			Key: "active",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65545, 65546, 65547},
	},
	{
		Key: 65545,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active",
				Summary: "Amount of guest physical memory that is being actively read or written by guest. Activeness is estimated by ESXi",
			},
			Key: "active",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65546,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active",
				Summary: "Amount of guest physical memory that is being actively read or written by guest. Activeness is estimated by ESXi",
			},
			Key: "active",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65547,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active",
				Summary: "Amount of guest physical memory that is being actively read or written by guest. Activeness is estimated by ESXi",
			},
			Key: "active",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65548,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared",
				Summary: "Amount of guest physical memory that is shared within a single virtual machine or across virtual machines",
			},
			Key: "shared",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65549, 65550, 65551},
	},
	{
		Key: 65549,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared",
				Summary: "Amount of guest physical memory that is shared within a single virtual machine or across virtual machines",
			},
			Key: "shared",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65550,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared",
				Summary: "Amount of guest physical memory that is shared within a single virtual machine or across virtual machines",
			},
			Key: "shared",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65551,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared",
				Summary: "Amount of guest physical memory that is shared within a single virtual machine or across virtual machines",
			},
			Key: "shared",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65552,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Zero pages",
				Summary: "Guest physical memory pages whose content is 0x00",
			},
			Key: "zero",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65553, 65554, 65555},
	},
	{
		Key: 65553,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Zero pages",
				Summary: "Guest physical memory pages whose content is 0x00",
			},
			Key: "zero",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65554,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Zero pages",
				Summary: "Guest physical memory pages whose content is 0x00",
			},
			Key: "zero",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65555,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Zero pages",
				Summary: "Guest physical memory pages whose content is 0x00",
			},
			Key: "zero",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65556,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reservation available",
				Summary: "Amount by which reservation can be raised",
			},
			Key: "unreserved",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65557, 65558, 65559},
	},
	{
		Key: 65557,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reservation available",
				Summary: "Amount by which reservation can be raised",
			},
			Key: "unreserved",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65558,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reservation available",
				Summary: "Amount by which reservation can be raised",
			},
			Key: "unreserved",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65559,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reservation available",
				Summary: "Amount by which reservation can be raised",
			},
			Key: "unreserved",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65560,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap consumed",
				Summary: "Swap storage space consumed",
			},
			Key: "swapused",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65561, 65562, 65563},
	},
	{
		Key: 65561,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap consumed",
				Summary: "Swap storage space consumed",
			},
			Key: "swapused",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65562,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap consumed",
				Summary: "Swap storage space consumed",
			},
			Key: "swapused",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65563,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap consumed",
				Summary: "Swap storage space consumed",
			},
			Key: "swapused",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65568,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared common",
				Summary: "Amount of host physical memory that backs shared guest physical memory (Shared)",
			},
			Key: "sharedcommon",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65569, 65570, 65571},
	},
	{
		Key: 65569,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared common",
				Summary: "Amount of host physical memory that backs shared guest physical memory (Shared)",
			},
			Key: "sharedcommon",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65570,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared common",
				Summary: "Amount of host physical memory that backs shared guest physical memory (Shared)",
			},
			Key: "sharedcommon",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65571,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Shared common",
				Summary: "Amount of host physical memory that backs shared guest physical memory (Shared)",
			},
			Key: "sharedcommon",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65572,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap",
				Summary: "Virtual address space of ESXi that is dedicated to its heap",
			},
			Key: "heap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65573, 65574, 65575},
	},
	{
		Key: 65573,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap",
				Summary: "Virtual address space of ESXi that is dedicated to its heap",
			},
			Key: "heap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65574,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap",
				Summary: "Virtual address space of ESXi that is dedicated to its heap",
			},
			Key: "heap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65575,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap",
				Summary: "Virtual address space of ESXi that is dedicated to its heap",
			},
			Key: "heap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65576,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap free",
				Summary: "Free address space in the heap of ESXi. This is less than or equal to Heap",
			},
			Key: "heapfree",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65577, 65578, 65579},
	},
	{
		Key: 65577,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap free",
				Summary: "Free address space in the heap of ESXi. This is less than or equal to Heap",
			},
			Key: "heapfree",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65578,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap free",
				Summary: "Free address space in the heap of ESXi. This is less than or equal to Heap",
			},
			Key: "heapfree",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65579,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heap free",
				Summary: "Free address space in the heap of ESXi. This is less than or equal to Heap",
			},
			Key: "heapfree",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65580,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Free state",
				Summary: "Current memory availability state of ESXi. Possible values are high, clear, soft, hard, low. The state value determines the techniques used for memory reclamation from virtual machines",
			},
			Key: "state",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65581,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Ballooned memory",
				Summary: "Amount of guest physical memory reclaimed from the virtual machine by the balloon driver in the guest",
			},
			Key: "vmmemctl",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65582, 65583, 65584},
	},
	{
		Key: 65582,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Ballooned memory",
				Summary: "Amount of guest physical memory reclaimed from the virtual machine by the balloon driver in the guest",
			},
			Key: "vmmemctl",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65583,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Ballooned memory",
				Summary: "Amount of guest physical memory reclaimed from the virtual machine by the balloon driver in the guest",
			},
			Key: "vmmemctl",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65584,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Ballooned memory",
				Summary: "Amount of guest physical memory reclaimed from the virtual machine by the balloon driver in the guest",
			},
			Key: "vmmemctl",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65585,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead consumed",
				Summary: "Host physical memory consumed by ESXi data structures for running the virtual machines",
			},
			Key: "overhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65586, 65587, 65588},
	},
	{
		Key: 65586,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead consumed",
				Summary: "Host physical memory consumed by ESXi data structures for running the virtual machines",
			},
			Key: "overhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65587,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead consumed",
				Summary: "Host physical memory consumed by ESXi data structures for running the virtual machines",
			},
			Key: "overhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65588,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead consumed",
				Summary: "Host physical memory consumed by ESXi data structures for running the virtual machines",
			},
			Key: "overhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65589,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reservation consumed",
				Summary: "Memory reservation consumed by powered-on virtual machines",
			},
			Key: "reservedCapacity",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MB",
				Summary: "Megabyte",
			},
			Key: "megaBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65590,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swapped",
				Summary: "Amount of guest physical memory that is swapped out to the swap space",
			},
			Key: "swapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65591, 65592, 65593},
	},
	{
		Key: 65591,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swapped",
				Summary: "Amount of guest physical memory that is swapped out to the swap space",
			},
			Key: "swapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65592,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swapped",
				Summary: "Amount of guest physical memory that is swapped out to the swap space",
			},
			Key: "swapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65593,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swapped",
				Summary: "Amount of guest physical memory that is swapped out to the swap space",
			},
			Key: "swapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65594,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap target",
				Summary: "Amount of memory that ESXi needs to reclaim by swapping",
			},
			Key: "swaptarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65595, 65596, 65597},
	},
	{
		Key: 65595,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap target",
				Summary: "Amount of memory that ESXi needs to reclaim by swapping",
			},
			Key: "swaptarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65596,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap target",
				Summary: "Amount of memory that ESXi needs to reclaim by swapping",
			},
			Key: "swaptarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65597,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap target",
				Summary: "Amount of memory that ESXi needs to reclaim by swapping",
			},
			Key: "swaptarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65598,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap in",
				Summary: "Amount of guest physical memory that is swapped in from the swap space since the virtual machine has been powered on. This value is less than or equal to the 'Swap out' counter",
			},
			Key: "swapin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65599, 65600, 65601},
	},
	{
		Key: 65599,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap in",
				Summary: "Amount of guest physical memory that is swapped in from the swap space since the virtual machine has been powered on. This value is less than or equal to the 'Swap out' counter",
			},
			Key: "swapin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65600,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap in",
				Summary: "Amount of guest physical memory that is swapped in from the swap space since the virtual machine has been powered on. This value is less than or equal to the 'Swap out' counter",
			},
			Key: "swapin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65601,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap in",
				Summary: "Amount of guest physical memory that is swapped in from the swap space since the virtual machine has been powered on. This value is less than or equal to the 'Swap out' counter",
			},
			Key: "swapin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65602,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap out",
				Summary: "Amount of guest physical memory that is swapped out from the virtual machine to its swap space since it has been powered on.",
			},
			Key: "swapout",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65603, 65604, 65605},
	},
	{
		Key: 65603,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap out",
				Summary: "Amount of guest physical memory that is swapped out from the virtual machine to its swap space since it has been powered on.",
			},
			Key: "swapout",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65604,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap out",
				Summary: "Amount of guest physical memory that is swapped out from the virtual machine to its swap space since it has been powered on.",
			},
			Key: "swapout",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65605,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap out",
				Summary: "Amount of guest physical memory that is swapped out from the virtual machine to its swap space since it has been powered on.",
			},
			Key: "swapout",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65606,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Balloon target",
				Summary: "Desired amount of guest physical memory the balloon driver needs to reclaim, as determined by ESXi",
			},
			Key: "vmmemctltarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65607, 65608, 65609},
	},
	{
		Key: 65607,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Balloon target",
				Summary: "Desired amount of guest physical memory the balloon driver needs to reclaim, as determined by ESXi",
			},
			Key: "vmmemctltarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65608,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Balloon target",
				Summary: "Desired amount of guest physical memory the balloon driver needs to reclaim, as determined by ESXi",
			},
			Key: "vmmemctltarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65609,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Balloon target",
				Summary: "Desired amount of guest physical memory the balloon driver needs to reclaim, as determined by ESXi",
			},
			Key: "vmmemctltarget",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65610,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Consumed",
				Summary: "Amount of host physical memory consumed for backing up guest physical memory pages",
			},
			Key: "consumed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65611, 65612, 65613},
	},
	{
		Key: 65611,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Consumed",
				Summary: "Amount of host physical memory consumed for backing up guest physical memory pages",
			},
			Key: "consumed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65612,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Consumed",
				Summary: "Amount of host physical memory consumed for backing up guest physical memory pages",
			},
			Key: "consumed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65613,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Consumed",
				Summary: "Amount of host physical memory consumed for backing up guest physical memory pages",
			},
			Key: "consumed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65614,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMkernel consumed",
				Summary: "Amount of host physical memory consumed by VMkernel",
			},
			Key: "sysUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65615, 65616, 65617},
	},
	{
		Key: 65615,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMkernel consumed",
				Summary: "Amount of host physical memory consumed by VMkernel",
			},
			Key: "sysUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65616,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMkernel consumed",
				Summary: "Amount of host physical memory consumed by VMkernel",
			},
			Key: "sysUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65617,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMkernel consumed",
				Summary: "Amount of host physical memory consumed by VMkernel",
			},
			Key: "sysUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65618,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap in rate",
				Summary: "Rate at which guest physical memory is swapped in from the swap space",
			},
			Key: "swapinRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65619,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Swap out rate",
				Summary: "Rate at which guest physical memory is swapped out to the swap space",
			},
			Key: "swapoutRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65620,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active write",
				Summary: "Amount of guest physical memory that is being actively written by guest. Activeness is estimated by ESXi",
			},
			Key: "activewrite",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65621,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Compressed",
				Summary: "Guest physical memory pages that have undergone memory compression",
			},
			Key: "compressed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65622,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Compression rate",
				Summary: "Rate of guest physical memory page compression by ESXi",
			},
			Key: "compressionRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65623,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Decompression rate",
				Summary: "Rate of guest physical memory decompression",
			},
			Key: "decompressionRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65624,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead reserved",
				Summary: "Host physical memory reserved by ESXi, for its data structures, for running the virtual machine",
			},
			Key: "overheadMax",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65625,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Total reservation",
				Summary: "Total reservation, available and consumed, for powered-on virtual machines",
			},
			Key: "totalCapacity",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MB",
				Summary: "Megabyte",
			},
			Key: "megaBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65626,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Compressed",
				Summary: "Amount of guest physical memory pages compressed by ESXi",
			},
			Key: "zipped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65627,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Compression saved",
				Summary: "Host physical memory, reclaimed from a virtual machine, by memory compression. This value is less than the value of 'Compressed' memory",
			},
			Key: "zipSaved",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65628,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Page-fault latency",
				Summary: "Percentage of time the virtual machine spent waiting to swap in or decompress guest physical memory",
			},
			Key: "latency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65629,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Entitlement",
				Summary: "Amount of host physical memory the virtual machine deserves, as determined by ESXi",
			},
			Key: "entitlement",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65630,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Reclamation threshold",
				Summary: "Threshold of free host physical memory below which ESXi will begin actively reclaiming memory from virtual machines by swapping, compression and ballooning",
			},
			Key: "lowfreethreshold",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65631,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache consumed",
				Summary: "Storage space consumed on the host swap cache for storing swapped guest physical memory pages",
			},
			Key: "llSwapUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65635, 65636, 65637},
	},
	{
		Key: 65632,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap in rate",
				Summary: "Rate at which guest physical memory is swapped in from the host swap cache",
			},
			Key: "llSwapInRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65633,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap out rate",
				Summary: "Rate at which guest physical memory is swapped out to the host swap cache",
			},
			Key: "llSwapOutRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65634,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead active",
				Summary: "Estimate of the host physical memory, from Overhead consumed, that is actively read or written to by ESXi",
			},
			Key: "overheadTouched",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65635,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache consumed",
				Summary: "Storage space consumed on the host swap cache for storing swapped guest physical memory pages",
			},
			Key: "llSwapUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65636,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache consumed",
				Summary: "Storage space consumed on the host swap cache for storing swapped guest physical memory pages",
			},
			Key: "llSwapUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65637,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache consumed",
				Summary: "Storage space consumed on the host swap cache for storing swapped guest physical memory pages",
			},
			Key: "llSwapUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65638,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap in",
				Summary: "Amount of guest physical memory swapped in from host cache",
			},
			Key: "llSwapIn",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65639, 65640, 65641},
	},
	{
		Key: 65639,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap in",
				Summary: "Amount of guest physical memory swapped in from host cache",
			},
			Key: "llSwapIn",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65640,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap in",
				Summary: "Amount of guest physical memory swapped in from host cache",
			},
			Key: "llSwapIn",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65641,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap in",
				Summary: "Amount of guest physical memory swapped in from host cache",
			},
			Key: "llSwapIn",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65642,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap out",
				Summary: "Amount of guest physical memory swapped out to the host swap cache",
			},
			Key: "llSwapOut",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{65643, 65644, 65645},
	},
	{
		Key: 65643,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap out",
				Summary: "Amount of guest physical memory swapped out to the host swap cache",
			},
			Key: "llSwapOut",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65644,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap out",
				Summary: "Amount of guest physical memory swapped out to the host swap cache",
			},
			Key: "llSwapOut",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65645,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host cache swap out",
				Summary: "Amount of guest physical memory swapped out to the host swap cache",
			},
			Key: "llSwapOut",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65646,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMFS PB Cache Size",
				Summary: "Space used for holding VMFS Pointer Blocks in memory",
			},
			Key: "vmfs.pbc.size",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MB",
				Summary: "Megabyte",
			},
			Key: "megaBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65647,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Maximum VMFS PB Cache Size",
				Summary: "Maximum size the VMFS Pointer Block Cache can grow to",
			},
			Key: "vmfs.pbc.sizeMax",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MB",
				Summary: "Megabyte",
			},
			Key: "megaBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65648,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMFS Working Set",
				Summary: "Amount of file blocks whose addresses are cached in the VMFS PB Cache",
			},
			Key: "vmfs.pbc.workingSet",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "TB",
				Summary: "Terabyte",
			},
			Key: "teraBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65649,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Maximum VMFS Working Set",
				Summary: "Maximum amount of file blocks whose addresses are cached in the VMFS PB Cache",
			},
			Key: "vmfs.pbc.workingSetMax",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "TB",
				Summary: "Terabyte",
			},
			Key: "teraBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65650,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMFS PB Cache Overhead",
				Summary: "Amount of VMFS heap used by the VMFS PB Cache",
			},
			Key: "vmfs.pbc.overhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 65651,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VMFS PB Cache Capacity Miss Ratio",
				Summary: "Trailing average of the ratio of capacity misses to compulsory misses for the VMFS PB Cache",
			},
			Key: "vmfs.pbc.capMissRatio",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory",
				Summary: "Memory",
			},
			Key: "mem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131072,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Aggregated disk I/O rate. For hosts, this metric includes the rates for all virtual machines running on the host during the collection interval.",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{131073, 131074, 131075},
	},
	{
		Key: 131073,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Aggregated disk I/O rate. For hosts, this metric includes the rates for all virtual machines running on the host during the collection interval.",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131074,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Aggregated disk I/O rate. For hosts, this metric includes the rates for all virtual machines running on the host during the collection interval.",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131075,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Aggregated disk I/O rate. For hosts, this metric includes the rates for all virtual machines running on the host during the collection interval.",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131076,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read requests",
				Summary: "Number of disk reads during the collection interval",
			},
			Key: "numberRead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131077,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write requests",
				Summary: "Number of disk writes during the collection interval",
			},
			Key: "numberWrite",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131078,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read rate",
				Summary: "Average number of kilobytes read from the disk each second during the collection interval",
			},
			Key: "read",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131079,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write rate",
				Summary: "Average number of kilobytes written to disk each second during the collection interval",
			},
			Key: "write",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131080,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Commands issued",
				Summary: "Number of SCSI commands issued during the collection interval",
			},
			Key: "commands",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131081,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Commands aborted",
				Summary: "Number of SCSI commands aborted during the collection interval",
			},
			Key: "commandsAborted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131082,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Bus resets",
				Summary: "Number of SCSI-bus reset commands issued during the collection interval",
			},
			Key: "busResets",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131083,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Physical device read latency",
				Summary: "Average amount of time, in milliseconds, to read from the physical device",
			},
			Key: "deviceReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131084,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Kernel read latency",
				Summary: "Average amount of time, in milliseconds, spent by VMkernel to process each SCSI read command",
			},
			Key: "kernelReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131085,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read latency",
				Summary: "Average amount of time taken during the collection interval to process a SCSI read command issued from the guest OS to the virtual machine",
			},
			Key: "totalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131086,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Queue read latency",
				Summary: "Average amount of time spent in the VMkernel queue, per SCSI read command, during the collection interval",
			},
			Key: "queueReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131087,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Physical device write latency",
				Summary: "Average amount of time, in milliseconds, to write to the physical device",
			},
			Key: "deviceWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131088,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Kernel write latency",
				Summary: "Average amount of time, in milliseconds, spent by VMkernel to process each SCSI write command",
			},
			Key: "kernelWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131089,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write latency",
				Summary: "Average amount of time taken during the collection interval to process a SCSI write command issued by the guest OS to the virtual machine",
			},
			Key: "totalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131090,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Queue write latency",
				Summary: "Average amount of time spent in the VMkernel queue, per SCSI write command, during the collection interval",
			},
			Key: "queueWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131091,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Physical device command latency",
				Summary: "Average amount of time, in milliseconds, to complete a SCSI command from the physical device",
			},
			Key: "deviceLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131092,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Kernel command latency",
				Summary: "Average amount of time, in milliseconds, spent by VMkernel to process each SCSI command",
			},
			Key: "kernelLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131093,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Command latency",
				Summary: "Average amount of time taken during the collection interval to process a SCSI command issued by the guest OS to the virtual machine",
			},
			Key: "totalLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131094,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Queue command latency",
				Summary: "Average amount of time spent in the VMkernel queue, per SCSI command, during the collection interval",
			},
			Key: "queueLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131095,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Highest latency",
				Summary: "Highest latency value across all disks used by the host",
			},
			Key: "maxTotalLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131096,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Maximum queue depth",
				Summary: "Maximum queue depth",
			},
			Key: "maxQueueDepth",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131097,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read requests per second",
				Summary: "Average number of disk reads per second during the collection interval",
			},
			Key: "numberReadAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131098,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write requests per second",
				Summary: "Average number of disk writes per second during the collection interval",
			},
			Key: "numberWriteAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 131099,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average commands issued per second",
				Summary: "Average number of SCSI commands issued per second during the collection interval",
			},
			Key: "commandsAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk",
				Summary: "Disk",
			},
			Key: "disk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196608,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Network utilization (combined transmit-rates and receive-rates) during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{196609, 196610, 196611},
	},
	{
		Key: 196609,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Network utilization (combined transmit-rates and receive-rates) during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196610,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Network utilization (combined transmit-rates and receive-rates) during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196611,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Network utilization (combined transmit-rates and receive-rates) during the interval",
			},
			Key: "usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196612,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Packets received",
				Summary: "Number of packets received during the interval",
			},
			Key: "packetsRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196613,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Packets transmitted",
				Summary: "Number of packets transmitted during the interval",
			},
			Key: "packetsTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196614,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Data receive rate",
				Summary: "Average rate at which data was received during the interval",
			},
			Key: "received",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196615,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Data transmit rate",
				Summary: "Average rate at which data was transmitted during the interval",
			},
			Key: "transmitted",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196616,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Receive packets dropped",
				Summary: "Number of receives dropped",
			},
			Key: "droppedRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196617,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Transmit packets dropped",
				Summary: "Number of transmits dropped",
			},
			Key: "droppedTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196618,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Data receive rate",
				Summary: "Average amount of data received per second",
			},
			Key: "bytesRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196619,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Data transmit rate",
				Summary: "Average amount of data transmitted per second",
			},
			Key: "bytesTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196620,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Broadcast receives",
				Summary: "Number of broadcast packets received during the sampling interval",
			},
			Key: "broadcastRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196621,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Broadcast transmits",
				Summary: "Number of broadcast packets transmitted during the sampling interval",
			},
			Key: "broadcastTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196622,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Multicast receives",
				Summary: "Number of multicast packets received during the sampling interval",
			},
			Key: "multicastRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196623,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Multicast transmits",
				Summary: "Number of multicast packets transmitted during the sampling interval",
			},
			Key: "multicastTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196624,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Packet receive errors",
				Summary: "Number of packets with errors received during the sampling interval",
			},
			Key: "errorsRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196625,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Packet transmit errors",
				Summary: "Number of packets with errors transmitted during the sampling interval",
			},
			Key: "errorsTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196626,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Unknown protocol frames",
				Summary: "Number of frames with unknown protocol received during the sampling interval",
			},
			Key: "unknownProtos",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196627,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pnicBytesRx",
				Summary: "pnicBytesRx",
			},
			Key: "pnicBytesRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 196628,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pnicBytesTx",
				Summary: "pnicBytesTx",
			},
			Key: "pnicBytesTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Network",
				Summary: "Network",
			},
			Key: "net",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262144,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Uptime",
				Summary: "Total time elapsed, in seconds, since last system startup",
			},
			Key: "uptime",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "s",
				Summary: "Second",
			},
			Key: "second",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262145,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Heartbeat",
				Summary: "Number of heartbeats issued per virtual machine during the interval",
			},
			Key: "heartbeat",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262146,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk usage",
				Summary: "Amount of disk space usage for each mount point",
			},
			Key: "diskUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262147,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU usage (None)",
				Summary: "Amount of CPU used by the Service Console and other applications during the interval",
			},
			Key: "resourceCpuUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "none",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{262148, 262149, 262150},
	},
	{
		Key: 262148,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU usage (Average)",
				Summary: "Amount of CPU used by the Service Console and other applications during the interval",
			},
			Key: "resourceCpuUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262149,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU usage (Maximum)",
				Summary: "Amount of CPU used by the Service Console and other applications during the interval",
			},
			Key: "resourceCpuUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "maximum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262150,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU usage (Minimum)",
				Summary: "Amount of CPU used by the Service Console and other applications during the interval",
			},
			Key: "resourceCpuUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "minimum",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262151,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory touched",
				Summary: "Memory touched by the system resource group",
			},
			Key: "resourceMemTouched",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262152,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory mapped",
				Summary: "Memory mapped by the system resource group",
			},
			Key: "resourceMemMapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262153,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory share saved",
				Summary: "Memory saved due to sharing by the system resource group",
			},
			Key: "resourceMemShared",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262154,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory swapped",
				Summary: "Memory swapped out by the system resource group",
			},
			Key: "resourceMemSwapped",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262155,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory overhead",
				Summary: "Overhead memory consumed by the system resource group",
			},
			Key: "resourceMemOverhead",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262156,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory shared",
				Summary: "Memory shared by the system resource group",
			},
			Key: "resourceMemCow",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262157,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory zero",
				Summary: "Zero filled memory used by the system resource group",
			},
			Key: "resourceMemZero",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262158,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU running (1 min. average)",
				Summary: "CPU running average over 1 minute of the system resource group",
			},
			Key: "resourceCpuRun1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262159,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU active (1 min average)",
				Summary: "CPU active average over 1 minute of the system resource group",
			},
			Key: "resourceCpuAct1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262160,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU maximum limited (1 min)",
				Summary: "CPU maximum limited over 1 minute of the system resource group",
			},
			Key: "resourceCpuMaxLimited1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262161,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU running (5 min average)",
				Summary: "CPU running average over 5 minutes of the system resource group",
			},
			Key: "resourceCpuRun5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262162,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU active (5 min average)",
				Summary: "CPU active average over 5 minutes of the system resource group",
			},
			Key: "resourceCpuAct5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262163,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU maximum limited (5 min)",
				Summary: "CPU maximum limited over 5 minutes of the system resource group",
			},
			Key: "resourceCpuMaxLimited5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262164,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU allocation minimum (in MHz)",
				Summary: "CPU allocation reservation (in MHz) of the system resource group",
			},
			Key: "resourceCpuAllocMin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262165,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU allocation maximum (in MHz)",
				Summary: "CPU allocation limit (in MHz) of the system resource group",
			},
			Key: "resourceCpuAllocMax",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262166,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource CPU allocation shares",
				Summary: "CPU allocation shares of the system resource group",
			},
			Key: "resourceCpuAllocShares",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262167,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory allocation minimum (in KB)",
				Summary: "Memory allocation reservation (in KB) of the system resource group",
			},
			Key: "resourceMemAllocMin",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262168,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory allocation maximum (in KB)",
				Summary: "Memory allocation limit (in KB) of the system resource group",
			},
			Key: "resourceMemAllocMax",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262169,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory allocation shares",
				Summary: "Memory allocation shares of the system resource group",
			},
			Key: "resourceMemAllocShares",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262170,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "OS Uptime",
				Summary: "Total time elapsed, in seconds, since last operating system boot-up",
			},
			Key: "osUptime",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "s",
				Summary: "Second",
			},
			Key: "second",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262171,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource memory consumed",
				Summary: "Memory consumed by the system resource group",
			},
			Key: "resourceMemConsumed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 262172,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "File descriptors used",
				Summary: "Number of file descriptors used by the system resource group",
			},
			Key: "resourceFdUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System",
				Summary: "System",
			},
			Key: "sys",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327680,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (1 min average)",
				Summary: "CPU active average over 1 minute",
			},
			Key: "actav1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327681,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (1 min peak)",
				Summary: "CPU active peak over 1 minute",
			},
			Key: "actpk1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327682,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (1 min average)",
				Summary: "CPU running average over 1 minute",
			},
			Key: "runav1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327683,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (5 min average)",
				Summary: "CPU active average over 5 minutes",
			},
			Key: "actav5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327684,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (5 min peak)",
				Summary: "CPU active peak over 5 minutes",
			},
			Key: "actpk5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327685,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (5 min average)",
				Summary: "CPU running average over 5 minutes",
			},
			Key: "runav5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327686,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (15 min average)",
				Summary: "CPU active average over 15 minutes",
			},
			Key: "actav15",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327687,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Active (15 min peak)",
				Summary: "CPU active peak over 15 minutes",
			},
			Key: "actpk15",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327688,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (15 min average)",
				Summary: "CPU running average over 15 minutes",
			},
			Key: "runav15",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327689,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (1 min peak)",
				Summary: "CPU running peak over 1 minute",
			},
			Key: "runpk1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327690,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Throttled (1 min average)",
				Summary: "Amount of CPU resources over the limit that were refused, average over 1 minute",
			},
			Key: "maxLimited1",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327691,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (5 min peak)",
				Summary: "CPU running peak over 5 minutes",
			},
			Key: "runpk5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327692,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Throttled (5 min average)",
				Summary: "Amount of CPU resources over the limit that were refused, average over 5 minutes",
			},
			Key: "maxLimited5",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327693,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Running (15 min peak)",
				Summary: "CPU running peak over 15 minutes",
			},
			Key: "runpk15",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327694,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Throttled (15 min average)",
				Summary: "Amount of CPU resources over the limit that were refused, average over 15 minutes",
			},
			Key: "maxLimited15",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327695,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Group CPU sample count",
				Summary: "Group CPU sample count",
			},
			Key: "sampleCount",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 327696,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Group CPU sample period",
				Summary: "Group CPU sample period",
			},
			Key: "samplePeriod",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Resource group CPU",
				Summary: "Resource group CPU",
			},
			Key: "rescpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 393216,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory used",
				Summary: "Amount of total configured memory that is available for use",
			},
			Key: "memUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Management agent",
				Summary: "Management agent",
			},
			Key: "managementAgent",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 393217,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory swap used",
				Summary: "Sum of the memory swapped by all powered-on virtual machines on the host",
			},
			Key: "swapUsed",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Management agent",
				Summary: "Management agent",
			},
			Key: "managementAgent",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 393218,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory swap in",
				Summary: "Amount of memory that is swapped in for the Service Console",
			},
			Key: "swapIn",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Management agent",
				Summary: "Management agent",
			},
			Key: "managementAgent",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 393219,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory swap out",
				Summary: "Amount of memory that is swapped out for the Service Console",
			},
			Key: "swapOut",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Management agent",
				Summary: "Management agent",
			},
			Key: "managementAgent",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 393220,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU usage",
				Summary: "Amount of Service Console CPU usage",
			},
			Key: "cpuUsage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Management agent",
				Summary: "Management agent",
			},
			Key: "managementAgent",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MHz",
				Summary: "Megahertz",
			},
			Key: "megaHertz",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458752,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average commands issued per second",
				Summary: "Average number of commands issued per second by the storage adapter during the collection interval",
			},
			Key: "commandsAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458753,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read requests per second",
				Summary: "Average number of read commands issued per second by the storage adapter during the collection interval",
			},
			Key: "numberReadAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458754,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write requests per second",
				Summary: "Average number of write commands issued per second by the storage adapter during the collection interval",
			},
			Key: "numberWriteAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458755,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read rate",
				Summary: "Rate of reading data by the storage adapter",
			},
			Key: "read",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458756,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write rate",
				Summary: "Rate of writing data by the storage adapter",
			},
			Key: "write",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458757,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read latency",
				Summary: "The average time a read by the storage adapter takes",
			},
			Key: "totalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458758,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write latency",
				Summary: "The average time a write by the storage adapter takes",
			},
			Key: "totalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 458759,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Highest latency",
				Summary: "Highest latency value across all storage adapters used by the host",
			},
			Key: "maxTotalLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage adapter",
				Summary: "Storage adapter",
			},
			Key: "storageAdapter",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524288,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average commands issued per second",
				Summary: "Average number of commands issued per second on the storage path during the collection interval",
			},
			Key: "commandsAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524289,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read requests per second",
				Summary: "Average number of read commands issued per second on the storage path during the collection interval",
			},
			Key: "numberReadAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524290,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write requests per second",
				Summary: "Average number of write commands issued per second on the storage path during the collection interval",
			},
			Key: "numberWriteAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524291,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read rate",
				Summary: "Rate of reading data on the storage path",
			},
			Key: "read",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524292,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write rate",
				Summary: "Rate of writing data on the storage path",
			},
			Key: "write",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524293,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read latency",
				Summary: "The average time a read issued on the storage path takes",
			},
			Key: "totalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524294,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write latency",
				Summary: "The average time a write issued on the storage path takes",
			},
			Key: "totalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 524295,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Highest latency",
				Summary: "Highest latency value across all storage paths used by the host",
			},
			Key: "maxTotalLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage path",
				Summary: "Storage path",
			},
			Key: "storagePath",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589824,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read requests per second",
				Summary: "Average number of read commands issued per second to the virtual disk during the collection interval",
			},
			Key: "numberReadAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589825,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write requests per second",
				Summary: "Average number of write commands issued per second to the virtual disk during the collection interval",
			},
			Key: "numberWriteAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589826,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read rate",
				Summary: "Rate of reading data from the virtual disk",
			},
			Key: "read",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589827,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write rate",
				Summary: "Rate of writing data to the virtual disk",
			},
			Key: "write",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589828,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read latency",
				Summary: "The average time a read from the virtual disk takes",
			},
			Key: "totalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589829,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write latency",
				Summary: "The average time a write to the virtual disk takes",
			},
			Key: "totalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589830,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average number of outstanding read requests",
				Summary: "Average number of outstanding read requests to the virtual disk during the collection interval",
			},
			Key: "readOIO",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589831,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average number of outstanding write requests",
				Summary: "Average number of outstanding write requests to the virtual disk during the collection interval",
			},
			Key: "writeOIO",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589832,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read workload metric",
				Summary: "Storage DRS virtual disk metric for the read workload model",
			},
			Key: "readLoadMetric",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589833,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write workload metric",
				Summary: "Storage DRS virtual disk metric for the write workload model",
			},
			Key: "writeLoadMetric",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589834,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read request size",
				Summary: "Average read request size in bytes",
			},
			Key: "readIOSize",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589835,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write request size",
				Summary: "Average write request size in bytes",
			},
			Key: "writeIOSize",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589836,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Number of small seeks",
				Summary: "Number of seeks during the interval that were less than 64 LBNs apart",
			},
			Key: "smallSeeks",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589837,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Number of medium seeks",
				Summary: "Number of seeks during the interval that were between 64 and 8192 LBNs apart",
			},
			Key: "mediumSeeks",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589838,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Number of large seeks",
				Summary: "Number of seeks during the interval that were greater than 8192 LBNs apart",
			},
			Key: "largeSeeks",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589839,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read Latency (us)",
				Summary: "Read latency in microseconds",
			},
			Key: "readLatencyUS",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "µs",
				Summary: "Microsecond",
			},
			Key: "microsecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589840,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write Latency (us)",
				Summary: "Write latency in microseconds",
			},
			Key: "writeLatencyUS",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "µs",
				Summary: "Microsecond",
			},
			Key: "microsecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589841,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Flash Read Cache I/Os per second for the virtual disk",
				Summary: "The average virtual Flash Read Cache I/Os per second value for the virtual disk",
			},
			Key: "vFlashCacheIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589842,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Flash Read Cache latency for the virtual disk",
				Summary: "The average virtual Flash Read Cache latency value for the virtual disk",
			},
			Key: "vFlashCacheLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "µs",
				Summary: "Microsecond",
			},
			Key: "microsecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 589843,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Flash Read Cache throughput for virtual disk",
				Summary: "The average virtual Flash Read Cache throughput value for the virtual disk",
			},
			Key: "vFlashCacheThroughput",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual disk",
				Summary: "Virtual disk",
			},
			Key: "virtualDisk",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655360,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read requests per second",
				Summary: "Average number of read commands issued per second to the datastore during the collection interval",
			},
			Key: "numberReadAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655361,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write requests per second",
				Summary: "Average number of write commands issued per second to the datastore during the collection interval",
			},
			Key: "numberWriteAveraged",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655362,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read rate",
				Summary: "Rate of reading data from the datastore",
			},
			Key: "read",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655363,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write rate",
				Summary: "Rate of writing data to the datastore",
			},
			Key: "write",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655364,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read latency",
				Summary: "The average time a read from the datastore takes",
			},
			Key: "totalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655365,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write latency",
				Summary: "The average time a write to the datastore takes",
			},
			Key: "totalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655366,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage I/O Control normalized latency",
				Summary: "Storage I/O Control size-normalized I/O latency",
			},
			Key: "sizeNormalizedDatastoreLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "µs",
				Summary: "Microsecond",
			},
			Key: "microsecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655367,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage I/O Control aggregated IOPS",
				Summary: "Storage I/O Control aggregated IOPS",
			},
			Key: "datastoreIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655368,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore bytes read",
				Summary: "Storage DRS datastore bytes read",
			},
			Key: "datastoreReadBytes",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655369,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore bytes written",
				Summary: "Storage DRS datastore bytes written",
			},
			Key: "datastoreWriteBytes",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655370,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore read I/O rate",
				Summary: "Storage DRS datastore read I/O rate",
			},
			Key: "datastoreReadIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655371,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore write I/O rate",
				Summary: "Storage DRS datastore write I/O rate",
			},
			Key: "datastoreWriteIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655372,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore normalized read latency",
				Summary: "Storage DRS datastore normalized read latency",
			},
			Key: "datastoreNormalReadLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655373,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore normalized write latency",
				Summary: "Storage DRS datastore normalized write latency",
			},
			Key: "datastoreNormalWriteLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655374,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore outstanding read requests",
				Summary: "Storage DRS datastore outstanding read requests",
			},
			Key: "datastoreReadOIO",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655375,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore outstanding write requests",
				Summary: "Storage DRS datastore outstanding write requests",
			},
			Key: "datastoreWriteOIO",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655376,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage I/O Control datastore maximum queue depth",
				Summary: "Storage I/O Control datastore maximum queue depth",
			},
			Key: "datastoreMaxQueueDepth",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655377,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore read workload metric",
				Summary: "Storage DRS datastore metric for read workload model",
			},
			Key: "datastoreReadLoadMetric",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655378,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage DRS datastore write workload metric",
				Summary: "Storage DRS datastore metric for write workload model",
			},
			Key: "datastoreWriteLoadMetric",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655379,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Highest latency",
				Summary: "Highest latency value across all datastores used by the host",
			},
			Key: "maxTotalLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655380,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage I/O Control active time percentage",
				Summary: "Percentage of time Storage I/O Control actively controlled datastore latency",
			},
			Key: "siocActiveTimePercentage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 655381,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore latency observed by VMs",
				Summary: "The average datastore latency as seen by virtual machines",
			},
			Key: "datastoreVMObservedLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Datastore",
				Summary: "Datastore",
			},
			Key: "datastore",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "µs",
				Summary: "Microsecond",
			},
			Key: "microsecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 720896,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Usage",
				Summary: "Current power usage",
			},
			Key: "power",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Power",
				Summary: "Power",
			},
			Key: "power",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "W",
				Summary: "Watt",
			},
			Key: "watt",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 720897,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cap",
				Summary: "Maximum allowed power usage",
			},
			Key: "powerCap",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Power",
				Summary: "Power",
			},
			Key: "power",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "W",
				Summary: "Watt",
			},
			Key: "watt",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 720898,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Energy usage",
				Summary: "Total energy used since last stats reset",
			},
			Key: "energy",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Power",
				Summary: "Power",
			},
			Key: "power",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "J",
				Summary: "Joule",
			},
			Key: "joule",
		},
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 786432,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Replication VM Count",
				Summary: "Current number of replicated virtual machines",
			},
			Key: "hbrNumVms",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Replication",
				Summary: "vSphere Replication",
			},
			Key: "hbr",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 786433,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Replication Data Receive Rate",
				Summary: "Average amount of data received per second",
			},
			Key: "hbrNetRx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Replication",
				Summary: "vSphere Replication",
			},
			Key: "hbr",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 786434,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Replication Data Transmit Rate",
				Summary: "Average amount of data transmitted per second",
			},
			Key: "hbrNetTx",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Replication",
				Summary: "vSphere Replication",
			},
			Key: "hbr",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 851968,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Number of caches controlled by the virtual flash module",
				Summary: "Number of caches controlled by the virtual flash module",
			},
			Key: "numActiveVMDKs",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual flash",
				Summary: "Virtual flash module related statistical values",
			},
			Key: "vflashModule",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245184,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read IOPS",
				Summary: "Read IOPS",
			},
			Key: "readIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245185,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read throughput",
				Summary: "Read throughput in kBps",
			},
			Key: "readThroughput",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245186,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average read latency",
				Summary: "Average read latency in ms",
			},
			Key: "readAvgLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245187,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Max read latency",
				Summary: "Max read latency in ms",
			},
			Key: "readMaxLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245188,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cache hit rate",
				Summary: "Cache hit rate percentage",
			},
			Key: "readCacheHitRate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245189,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Read congestion per sampling interval",
				Summary: "Read congestion",
			},
			Key: "readCongestion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245190,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write IOPS",
				Summary: "Write IOPS",
			},
			Key: "writeIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245191,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write throughput",
				Summary: "Write throughput in kBps",
			},
			Key: "writeThroughput",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245192,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average write latency",
				Summary: "Average write latency in ms",
			},
			Key: "writeAvgLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245193,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Max write latency",
				Summary: "Max write latency in ms",
			},
			Key: "writeMaxLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245194,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Write congestion per sampling interval",
				Summary: "Write congestion",
			},
			Key: "writeCongestion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245195,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Recovery write IOPS",
				Summary: "Recovery write IOPS",
			},
			Key: "recoveryWriteIops",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245196,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Recovery write through-put",
				Summary: "Recovery write through-put in kBps",
			},
			Key: "recoveryWriteThroughput",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KBps",
				Summary: "Kilobytes per second",
			},
			Key: "kiloBytesPerSecond",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245197,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Average recovery write latency",
				Summary: "Average recovery write latency in ms",
			},
			Key: "recoveryWriteAvgLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245198,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Max recovery write latency",
				Summary: "Max recovery write latency in ms",
			},
			Key: "recoveryWriteMaxLatency",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1245199,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Recovery write congestion per sampling interval",
				Summary: "Recovery write congestion",
			},
			Key: "recoveryWriteCongestion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VSAN DOM Objects",
				Summary: "VSAN DOM object related statistical values",
			},
			Key: "vsanDomObj",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "rate",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310720,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "The utilization of a GPU in percentages",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{1310721, 1310722, 1310723},
	},
	{
		Key: 1310721,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "The utilization of a GPU in percentages",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310722,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "The utilization of a GPU in percentages",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310723,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Utilization",
				Summary: "The utilization of a GPU in percentages",
			},
			Key: "utilization",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310724,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory used",
				Summary: "The amount of GPU memory used in kilobytes",
			},
			Key: "mem.used",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{1310725, 1310726, 1310727},
	},
	{
		Key: 1310725,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory used",
				Summary: "The amount of GPU memory used in kilobytes",
			},
			Key: "mem.used",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310726,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory used",
				Summary: "The amount of GPU memory used in kilobytes",
			},
			Key: "mem.used",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310727,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory used",
				Summary: "The amount of GPU memory used in kilobytes",
			},
			Key: "mem.used",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310728,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory usage",
				Summary: "The amount of GPU memory used in percentages of the total available",
			},
			Key: "mem.usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "none",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: []int32{1310729, 1310730, 1310731},
	},
	{
		Key: 1310729,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory usage",
				Summary: "The amount of GPU memory used in percentages of the total available",
			},
			Key: "mem.usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310730,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory usage",
				Summary: "The amount of GPU memory used in percentages of the total available",
			},
			Key: "mem.usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310731,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory usage",
				Summary: "The amount of GPU memory used in percentages of the total available",
			},
			Key: "mem.usage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1310732,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Temperature",
				Summary: "The temperature of a GPU in degrees celsius",
			},
			Key: "temperature",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "GPU",
				Summary: "GPU",
			},
			Key: "gpu",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "℃",
				Summary: "Temperature in degrees Celsius",
			},
			Key: "celsius",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
	{
		Key: 1376256,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Persistent memory available reservation",
				Summary: "Persistent memory available reservation on a host.",
			},
			Key: "available.reservation",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "PMEM",
				Summary: "PMEM",
			},
			Key: "pmem",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "MB",
				Summary: "Megabyte",
			},
			Key: "megaBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               0,
		PerDeviceLevel:      0,
		AssociatedCounterId: nil,
	},
}

// *********************************** VM Metrics ************************************
var VmMetrics = []types.PerfMetricId{
	{
		CounterId: 11,
		Instance:  "$cpu",
	},
	{
		CounterId: 1,
		Instance:  "",
	},
	{
		CounterId: 12,
		Instance:  "",
	},
	{
		CounterId: 9,
		Instance:  "",
	},
	{
		CounterId: 29,
		Instance:  "",
	},
	{
		CounterId: 30,
		Instance:  "$cpu",
	},
	{
		CounterId: 24,
		Instance:  "",
	},
	{
		CounterId: 13,
		Instance:  "",
	},
	{
		CounterId: 10,
		Instance:  "$cpu",
	},
	{
		CounterId: 14,
		Instance:  "$cpu",
	},
	{
		CounterId: 27,
		Instance:  "$cpu",
	},
	{
		CounterId: 25,
		Instance:  "",
	},
	{
		CounterId: 5,
		Instance:  "$cpu",
	},
	{
		CounterId: 32,
		Instance:  "$cpu",
	},
	{
		CounterId: 14,
		Instance:  "",
	},
	{
		CounterId: 12,
		Instance:  "$cpu",
	},
	{
		CounterId: 10,
		Instance:  "",
	},
	{
		CounterId: 28,
		Instance:  "$cpu",
	},
	{
		CounterId: 5,
		Instance:  "",
	},
	{
		CounterId: 27,
		Instance:  "",
	},
	{
		CounterId: 31,
		Instance:  "",
	},
	{
		CounterId: 32,
		Instance:  "",
	},
	{
		CounterId: 26,
		Instance:  "",
	},
	{
		CounterId: 13,
		Instance:  "$cpu",
	},
	{
		CounterId: 28,
		Instance:  "",
	},
	{
		CounterId: 30,
		Instance:  "",
	},
	{
		CounterId: 11,
		Instance:  "",
	},
	{
		CounterId: 655379,
		Instance:  "",
	},

	{
		CounterId: 655362,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655363,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655360,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655364,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655361,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655365,
		Instance:  "$physDisk",
	},

	{
		CounterId: 131095,
		Instance:  "",
	},
	{
		CounterId: 65549,
		Instance:  "",
	},
	{
		CounterId: 65595,
		Instance:  "",
	},
	{
		CounterId: 65632,
		Instance:  "",
	},
	{
		CounterId: 65591,
		Instance:  "",
	},
	{
		CounterId: 65623,
		Instance:  "",
	},
	{
		CounterId: 65628,
		Instance:  "",
	},
	{
		CounterId: 65621,
		Instance:  "",
	},
	{
		CounterId: 65618,
		Instance:  "",
	},
	{
		CounterId: 65634,
		Instance:  "",
	},
	{
		CounterId: 65624,
		Instance:  "",
	},
	{
		CounterId: 65586,
		Instance:  "",
	},
	{
		CounterId: 65545,
		Instance:  "",
	},
	{
		CounterId: 65633,
		Instance:  "",
	},
	{
		CounterId: 65607,
		Instance:  "",
	},
	{
		CounterId: 65541,
		Instance:  "",
	},
	{
		CounterId: 65626,
		Instance:  "",
	},
	{
		CounterId: 65620,
		Instance:  "",
	},
	{
		CounterId: 65611,
		Instance:  "",
	},
	{
		CounterId: 65629,
		Instance:  "",
	},
	{
		CounterId: 65622,
		Instance:  "",
	},
	{
		CounterId: 65619,
		Instance:  "",
	},
	{
		CounterId: 65553,
		Instance:  "",
	},
	{
		CounterId: 65627,
		Instance:  "",
	},
	{
		CounterId: 65635,
		Instance:  "",
	},
	{
		CounterId: 65599,
		Instance:  "",
	},
	{
		CounterId: 65582,
		Instance:  "",
	},
	{
		CounterId: 65537,
		Instance:  "",
	},
	{
		CounterId: 65603,
		Instance:  "",
	},
	{
		CounterId: 196622,
		Instance:  "4000",
	},
	{
		CounterId: 196612,
		Instance:  "",
	},
	{
		CounterId: 196617,
		Instance:  "",
	},
	{
		CounterId: 196613,
		Instance:  "",
	},
	{
		CounterId: 196619,
		Instance:  "4000",
	},
	{
		CounterId: 196618,
		Instance:  "4000",
	},
	{
		CounterId: 196617,
		Instance:  "4000",
	},
	{
		CounterId: 196621,
		Instance:  "4000",
	},
	{
		CounterId: 196616,
		Instance:  "4000",
	},
	{
		CounterId: 196615,
		Instance:  "4000",
	},
	{
		CounterId: 196614,
		Instance:  "4000",
	},
	{
		CounterId: 196618,
		Instance:  "",
	},
	{
		CounterId: 196609,
		Instance:  "4000",
	},
	{
		CounterId: 196619,
		Instance:  "",
	},
	{
		CounterId: 196622,
		Instance:  "",
	},
	{
		CounterId: 196628,
		Instance:  "4000",
	},
	{
		CounterId: 196609,
		Instance:  "",
	},
	{
		CounterId: 196612,
		Instance:  "4000",
	},
	{
		CounterId: 196628,
		Instance:  "",
	},
	{
		CounterId: 196627,
		Instance:  "",
	},
	{
		CounterId: 196616,
		Instance:  "",
	},
	{
		CounterId: 196613,
		Instance:  "4000",
	},
	{
		CounterId: 196627,
		Instance:  "4000",
	},
	{
		CounterId: 196614,
		Instance:  "",
	},
	{
		CounterId: 196621,
		Instance:  "",
	},
	{
		CounterId: 196620,
		Instance:  "4000",
	},
	{
		CounterId: 196620,
		Instance:  "",
	},
	{
		CounterId: 196623,
		Instance:  "",
	},
	{
		CounterId: 196615,
		Instance:  "",
	},
	{
		CounterId: 196623,
		Instance:  "4000",
	},
	{
		CounterId: 720898,
		Instance:  "",
	},
	{
		CounterId: 720896,
		Instance:  "",
	},
	{
		CounterId: 327684,
		Instance:  "",
	},
	{
		CounterId: 327687,
		Instance:  "",
	},
	{
		CounterId: 327693,
		Instance:  "",
	},
	{
		CounterId: 327680,
		Instance:  "",
	},
	{
		CounterId: 327685,
		Instance:  "",
	},
	{
		CounterId: 327694,
		Instance:  "",
	},
	{
		CounterId: 327686,
		Instance:  "",
	},
	{
		CounterId: 327692,
		Instance:  "",
	},
	{
		CounterId: 327688,
		Instance:  "",
	},
	{
		CounterId: 327695,
		Instance:  "",
	},
	{
		CounterId: 327689,
		Instance:  "",
	},
	{
		CounterId: 327681,
		Instance:  "",
	},
	{
		CounterId: 327696,
		Instance:  "",
	},
	{
		CounterId: 327683,
		Instance:  "",
	},
	{
		CounterId: 327691,
		Instance:  "",
	},
	{
		CounterId: 327690,
		Instance:  "",
	},
	{
		CounterId: 327682,
		Instance:  "",
	},
	{
		CounterId: 262144,
		Instance:  "",
	},
	{
		CounterId: 262145,
		Instance:  "",
	},
	{
		CounterId: 262170,
		Instance:  "",
	},
	{
		CounterId: 589827,
		Instance:  "",
	},
	{
		CounterId: 589826,
		Instance:  "",
	},
}

// **************************** Host metrics *********************************

var HostMetrics = []types.PerfMetricId{
	{
		CounterId: 23,
		Instance:  "",
	},
	{
		CounterId: 14,
		Instance:  "",
	},
	{
		CounterId: 1,
		Instance:  "",
	},
	{
		CounterId: 11,
		Instance:  "",
	},
	{
		CounterId: 20,
		Instance:  "$cpu",
	},
	{
		CounterId: 13,
		Instance:  "",
	},
	{
		CounterId: 5,
		Instance:  "",
	},
	{
		CounterId: 32,
		Instance:  "",
	},
	{
		CounterId: 26,
		Instance:  "",
	},
	{
		CounterId: 24,
		Instance:  "",
	},
	{
		CounterId: 16,
		Instance:  "$cpu",
	},
	{
		CounterId: 27,
		Instance:  "",
	},
	{
		CounterId: 16,
		Instance:  "",
	},

	{
		CounterId: 10,
		Instance:  "",
	},
	{
		CounterId: 12,
		Instance:  "",
	},
	{
		CounterId: 1,
		Instance:  "$cpu",
	},
	{
		CounterId: 12,
		Instance:  "$cpu",
	},
	{
		CounterId: 13,
		Instance:  "$cpu",
	},
	{
		CounterId: 8,
		Instance:  "",
	},
	{
		CounterId: 655380,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655370,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655377,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655379,
		Instance:  "",
	},
	{
		CounterId: 655375,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655378,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655372,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655369,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655373,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655362,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655374,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655368,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655365,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655366,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655367,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655371,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655361,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655376,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655363,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655360,
		Instance:  "$physDisk",
	},
	{
		CounterId: 655381,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131073,
		Instance:  "",
	},
	{
		CounterId: 131090,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131079,
		Instance:  "",
	},
	{
		CounterId: 131086,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131098,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131081,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131082,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131090,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131081,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131086,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131088,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131098,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131078,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131079,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131099,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131087,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131089,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131078,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131096,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131091,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131080,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131078,
		Instance:  "",
	},
	{
		CounterId: 131076,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131092,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131080,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131095,
		Instance:  "",
	},
	{
		CounterId: 131097,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131093,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131092,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131084,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131099,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131079,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131085,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131083,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131076,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131096,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131094,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131088,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131089,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131077,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131077,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131093,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131087,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131085,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131091,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131097,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131082,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131094,
		Instance:  "mpx.vmhba32:C0:T0:L0",
	},
	{
		CounterId: 131084,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131083,
		Instance:  "$physDisk",
	},
	{
		CounterId: 786433,
		Instance:  "",
	},
	{
		CounterId: 786434,
		Instance:  "",
	},
	{
		CounterId: 786432,
		Instance:  "",
	},
	{
		CounterId: 65573,
		Instance:  "",
	},
	{
		CounterId: 65618,
		Instance:  "",
	},
	{
		CounterId: 65632,
		Instance:  "",
	},
	{
		CounterId: 65623,
		Instance:  "",
	},
	{
		CounterId: 65582,
		Instance:  "",
	},
	{
		CounterId: 65611,
		Instance:  "",
	},
	{
		CounterId: 65541,
		Instance:  "",
	},
	{
		CounterId: 65586,
		Instance:  "",
	},
	{
		CounterId: 65621,
		Instance:  "",
	},
	{
		CounterId: 65561,
		Instance:  "",
	},
	{
		CounterId: 65569,
		Instance:  "",
	},
	{
		CounterId: 65580,
		Instance:  "",
	},
	{
		CounterId: 65553,
		Instance:  "",
	},
	{
		CounterId: 65646,
		Instance:  "",
	},
	{
		CounterId: 65603,
		Instance:  "",
	},
	{
		CounterId: 65647,
		Instance:  "",
	},
	{
		CounterId: 65628,
		Instance:  "",
	},
	{
		CounterId: 65557,
		Instance:  "",
	},
	{
		CounterId: 65635,
		Instance:  "",
	},
	{
		CounterId: 65589,
		Instance:  "",
	},
	{
		CounterId: 65643,
		Instance:  "",
	},
	{
		CounterId: 65545,
		Instance:  "",
	},
	{
		CounterId: 65537,
		Instance:  "",
	},
	{
		CounterId: 65622,
		Instance:  "",
	},
	{
		CounterId: 65639,
		Instance:  "",
	},
	{
		CounterId: 65599,
		Instance:  "",
	},
	{
		CounterId: 65633,
		Instance:  "",
	},
	{
		CounterId: 65650,
		Instance:  "",
	},
	{
		CounterId: 65649,
		Instance:  "",
	},
	{
		CounterId: 65615,
		Instance:  "",
	},
	{
		CounterId: 65577,
		Instance:  "",
	},
	{
		CounterId: 65648,
		Instance:  "",
	},
	{
		CounterId: 65619,
		Instance:  "",
	},
	{
		CounterId: 65630,
		Instance:  "",
	},
	{
		CounterId: 65651,
		Instance:  "",
	},
	{
		CounterId: 65620,
		Instance:  "",
	},
	{
		CounterId: 65625,
		Instance:  "",
	},
	{
		CounterId: 65549,
		Instance:  "",
	},
	{
		CounterId: 196616,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196612,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196621,
		Instance:  "",
	},
	{
		CounterId: 196618,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196609,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196622,
		Instance:  "",
	},
	{
		CounterId: 196623,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196626,
		Instance:  "",
	},
	{
		CounterId: 196614,
		Instance:  "",
	},
	{
		CounterId: 196616,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196615,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196621,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196622,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196614,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196620,
		Instance:  "",
	},
	{
		CounterId: 196622,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196617,
		Instance:  "",
	},
	{
		CounterId: 196616,
		Instance:  "",
	},
	{
		CounterId: 196613,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196614,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196625,
		Instance:  "",
	},
	{
		CounterId: 196609,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196624,
		Instance:  "",
	},
	{
		CounterId: 196619,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196625,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196617,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196619,
		Instance:  "",
	},
	{
		CounterId: 196618,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196626,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196612,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196613,
		Instance:  "",
	},
	{
		CounterId: 196621,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196615,
		Instance:  "",
	},
	{
		CounterId: 196620,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196612,
		Instance:  "",
	},
	{
		CounterId: 196624,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196617,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196625,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196618,
		Instance:  "",
	},
	{
		CounterId: 196623,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196623,
		Instance:  "",
	},
	{
		CounterId: 196609,
		Instance:  "",
	},
	{
		CounterId: 196613,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196620,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196619,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196624,
		Instance:  "vmnic0",
	},
	{
		CounterId: 196615,
		Instance:  "vmnic1",
	},
	{
		CounterId: 196626,
		Instance:  "vmnic1",
	},
	{
		CounterId: 720898,
		Instance:  "",
	},
	{
		CounterId: 720897,
		Instance:  "",
	},
	{
		CounterId: 720896,
		Instance:  "",
	},
	{
		CounterId: 327681,
		Instance:  "",
	},
	{
		CounterId: 327694,
		Instance:  "",
	},
	{
		CounterId: 327689,
		Instance:  "",
	},
	{
		CounterId: 327696,
		Instance:  "",
	},
	{
		CounterId: 327685,
		Instance:  "",
	},
	{
		CounterId: 327680,
		Instance:  "",
	},
	{
		CounterId: 327690,
		Instance:  "",
	},
	{
		CounterId: 327693,
		Instance:  "",
	},
	{
		CounterId: 327683,
		Instance:  "",
	},
	{
		CounterId: 327688,
		Instance:  "",
	},
	{
		CounterId: 327687,
		Instance:  "",
	},
	{
		CounterId: 327684,
		Instance:  "",
	},
	{
		CounterId: 327691,
		Instance:  "",
	},
	{
		CounterId: 327682,
		Instance:  "",
	},
	{
		CounterId: 327695,
		Instance:  "",
	},
	{
		CounterId: 327686,
		Instance:  "",
	},
	{
		CounterId: 327692,
		Instance:  "",
	},
	{
		CounterId: 458755,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458755,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458756,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458757,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458753,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458754,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458752,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458755,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458757,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458753,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458754,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458752,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458759,
		Instance:  "",
	},
	{
		CounterId: 458758,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458753,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458758,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458756,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458754,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458753,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458757,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458754,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458756,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458752,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458752,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458756,
		Instance:  "vmhba32",
	},
	{
		CounterId: 458755,
		Instance:  "vmhba0",
	},
	{
		CounterId: 458758,
		Instance:  "vmhba64",
	},
	{
		CounterId: 458757,
		Instance:  "vmhba1",
	},
	{
		CounterId: 458758,
		Instance:  "vmhba32",
	},
	{
		CounterId: 524290,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524288,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524291,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524292,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524295,
		Instance:  "",
	},
	{
		CounterId: 524289,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524293,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524294,
		Instance:  "$physDisk",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262168,
		Instance:  "host/system",
	},
	{
		CounterId: 262172,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262166,
		Instance:  "host/system",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262163,
		Instance:  "host/system",
	},
	{
		CounterId: 262156,
		Instance:  "host/system",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262161,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262171,
		Instance:  "host",
	},
	{
		CounterId: 262156,
		Instance:  "host",
	},
	{
		CounterId: 262152,
		Instance:  "host",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262165,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262171,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262153,
		Instance:  "host",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262148,
		Instance:  "host/system",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262172,
		Instance:  "host",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262162,
		Instance:  "host/vim",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262153,
		Instance:  "host/system",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262154,
		Instance:  "host",
	},
	{
		CounterId: 262157,
		Instance:  "host/system",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262148,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262171,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262151,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262153,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262161,
		Instance:  "host/system",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262158,
		Instance:  "host/system",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262148,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262171,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262158,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262144,
		Instance:  "",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262164,
		Instance:  "host/system",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262163,
		Instance:  "host/vim",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262155,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262160,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262151,
		Instance:  "host/system",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262155,
		Instance:  "host/user",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262160,
		Instance:  "host/system",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262148,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262155,
		Instance:  "host/system",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262167,
		Instance:  "host/system",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262165,
		Instance:  "host/vim",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262159,
		Instance:  "host/system",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262157,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262163,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262151,
		Instance:  "host/user",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262155,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262154,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262171,
		Instance:  "host/system",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262153,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262157,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 262157,
		Instance:  "host",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262168,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262172,
		Instance:  "host/user",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262152,
		Instance:  "host/user",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262152,
		Instance:  "host/system",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262162,
		Instance:  "host/system",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262148,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262157,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262159,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262164,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262167,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262167,
		Instance:  "host/vim",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262169,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262166,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim",
	},
	{
		CounterId: 262165,
		Instance:  "host/system",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vsfwd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim",
	},
	{
		CounterId: 262169,
		Instance:  "host/system",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262158,
		Instance:  "host/vim",
	},
	{
		CounterId: 262159,
		Instance:  "host/vim",
	},
	{
		CounterId: 262160,
		Instance:  "host/vim",
	},
	{
		CounterId: 262161,
		Instance:  "host/vim",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262164,
		Instance:  "host/vim",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262166,
		Instance:  "host/vim",
	},
	{
		CounterId: 262168,
		Instance:  "host/vim",
	},
	{
		CounterId: 262169,
		Instance:  "host/vim",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262171,
		Instance:  "host/user",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 262148,
		Instance:  "host",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262155,
		Instance:  "host",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262162,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262155,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262172,
		Instance:  "host/system",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262172,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262154,
		Instance:  "host/system",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262151,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262156,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262152,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262155,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 262154,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262156,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262172,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 262151,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262152,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vsanmgmtdWatchdog",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262153,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 262154,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262155,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262157,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262151,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/pktcap-agent",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262153,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262152,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/vdpi",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262171,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262153,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262154,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262156,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 262154,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 262156,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 262157,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262171,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262172,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262152,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 262152,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 262151,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 262151,
		Instance:  "host",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 262148,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262148,
		Instance:  "host/user",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262154,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 262172,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262153,
		Instance:  "host/user",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262154,
		Instance:  "host/user",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262156,
		Instance:  "host/user",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262156,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262172,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 262157,
		Instance:  "host/user",
	},
	{
		CounterId: 262157,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262151,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262152,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262153,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262171,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 262155,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 262148,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 851968,
		Instance:  "vfc",
	},
}

// ********************************* Resource pool metrics **********************************
var ResourcePoolMetrics = []types.PerfMetricId{
	{
		CounterId: 5,
		Instance:  "",
	},
	{
		CounterId: 65586,
		Instance:  "",
	},
	{
		CounterId: 65591,
		Instance:  "",
	},
	{
		CounterId: 65545,
		Instance:  "",
	},
	{
		CounterId: 65553,
		Instance:  "",
	},
	{
		CounterId: 65541,
		Instance:  "",
	},
	{
		CounterId: 65549,
		Instance:  "",
	},
	{
		CounterId: 65582,
		Instance:  "",
	},
}
