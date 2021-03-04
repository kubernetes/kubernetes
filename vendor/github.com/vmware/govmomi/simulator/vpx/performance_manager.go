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

package vpx

import "github.com/vmware/govmomi/vim25/types"

// PerfCounter is the default template for the PerformanceManager perfCounter property.
// Capture method:
//   govc object.collect -s -dump PerformanceManager:PerfMgr perfCounter

var PerfCounter = []types.PerfCounterInfo{
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
		RollupType:          "none",
		StatsType:           "rate",
		Level:               4,
		PerDeviceLevel:      4,
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
		RollupType:          "average",
		StatsType:           "rate",
		Level:               1,
		PerDeviceLevel:      3,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 4,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
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
		RollupType:          "none",
		StatsType:           "rate",
		Level:               4,
		PerDeviceLevel:      4,
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
		RollupType:          "average",
		StatsType:           "rate",
		Level:               1,
		PerDeviceLevel:      3,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 8,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 9,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 10,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 11,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 12,
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
		Level:               1,
		PerDeviceLevel:      3,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 14,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 15,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Capacity Provisioned",
				Summary: "Capacity in MHz of the physical CPU cores",
			},
			Key: "capacity.provisioned",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 16,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Capacity Entitlement",
				Summary: "CPU resources devoted by the ESXi scheduler to the virtual machines and resource pools",
			},
			Key: "capacity.entitlement",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 17,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Capacity Usage",
				Summary: "CPU usage as a percent during the interval.",
			},
			Key: "capacity.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 18,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Capacity Demand",
				Summary: "The amount of CPU resources a VM would use if there were no CPU contention or CPU limit",
			},
			Key: "capacity.demand",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 19,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Capacity Contention",
				Summary: "Percent of time the VM is unable to run because it is contending for access to the physical CPU(s)",
			},
			Key: "capacity.contention",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 20,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Core Count Provisioned",
				Summary: "The number of virtual processors provisioned to the entity.",
			},
			Key: "corecount.provisioned",
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
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 21,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Core Count Usage",
				Summary: "The number of virtual processors running on the host.",
			},
			Key: "corecount.usage",
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
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 22,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU Core Count Contention",
				Summary: "Time the VM vCPU is ready to run, but is unable to run due to co-scheduling constraints",
			},
			Key: "corecount.contention",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 23,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 24,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 25,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 26,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 27,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 28,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 29,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 30,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 31,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 32,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 33,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 34,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 35,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 36,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 37,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 38,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 39,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 40,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 41,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 42,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 43,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 44,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 45,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 46,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 47,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 48,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 49,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 50,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 51,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 52,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapunreserved",
				Summary: "swapunreserved",
			},
			Key: "swapunreserved",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 53,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapunreserved",
				Summary: "swapunreserved",
			},
			Key: "swapunreserved",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 54,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapunreserved",
				Summary: "swapunreserved",
			},
			Key: "swapunreserved",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 55,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapunreserved",
				Summary: "swapunreserved",
			},
			Key: "swapunreserved",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 56,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 57,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 58,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 59,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 60,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 61,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 62,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 63,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 64,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 65,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 66,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 67,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 68,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 69,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 70,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 71,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 72,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 73,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 74,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 75,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 76,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 77,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapIn",
				Summary: "swapIn",
			},
			Key: "swapIn",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 78,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapIn",
				Summary: "swapIn",
			},
			Key: "swapIn",
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 79,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapIn",
				Summary: "swapIn",
			},
			Key: "swapIn",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 80,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapIn",
				Summary: "swapIn",
			},
			Key: "swapIn",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 81,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapOut",
				Summary: "swapOut",
			},
			Key: "swapOut",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 82,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapOut",
				Summary: "swapOut",
			},
			Key: "swapOut",
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 83,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapOut",
				Summary: "swapOut",
			},
			Key: "swapOut",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 84,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "swapOut",
				Summary: "swapOut",
			},
			Key: "swapOut",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 85,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 86,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 87,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 88,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 89,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 90,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 91,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 92,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 93,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 94,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 95,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 96,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 97,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 98,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 99,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 100,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 101,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 102,
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 103,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 104,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 105,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 106,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 107,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 108,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Capacity Provisioned",
				Summary: "Total amount of memory available to the host",
			},
			Key: "capacity.provisioned",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 109,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Capacity Entitlement",
				Summary: "Amount of host physical memory the VM is entitled to, as determined by the ESXi scheduler",
			},
			Key: "capacity.entitlement",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 110,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Capacity Usable",
				Summary: "Amount of physical memory available for use by virtual machines on this host",
			},
			Key: "capacity.usable",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 111,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Capacity Usage",
				Summary: "Amount of physical memory actively used",
			},
			Key: "capacity.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 112,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Capacity Contention",
				Summary: "Percentage of time VMs are waiting to access swapped, compressed or ballooned memory",
			},
			Key: "capacity.contention",
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
		StatsType:           "rate",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 113,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vm",
				Summary: "vm",
			},
			Key: "capacity.usage.vm",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 114,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vmOvrhd",
				Summary: "vmOvrhd",
			},
			Key: "capacity.usage.vmOvrhd",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 115,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vmkOvrhd",
				Summary: "vmkOvrhd",
			},
			Key: "capacity.usage.vmkOvrhd",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 116,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "userworld",
				Summary: "userworld",
			},
			Key: "capacity.usage.userworld",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 117,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vm",
				Summary: "vm",
			},
			Key: "reservedCapacity.vm",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 118,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vmOvhd",
				Summary: "vmOvhd",
			},
			Key: "reservedCapacity.vmOvhd",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 119,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vmkOvrhd",
				Summary: "vmkOvrhd",
			},
			Key: "reservedCapacity.vmkOvrhd",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 120,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "userworld",
				Summary: "userworld",
			},
			Key: "reservedCapacity.userworld",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 121,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Reserved Capacity %",
				Summary: "Percent of memory that has been reserved either through VMkernel use, by userworlds or due to VM memory reservations",
			},
			Key: "reservedCapacityPct",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 122,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Consumed by VMs",
				Summary: "Amount of physical memory consumed by VMs on this host",
			},
			Key: "consumed.vms",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 123,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory Consumed by userworlds",
				Summary: "Amount of physical memory consumed by userworlds on this host",
			},
			Key: "consumed.userworlds",
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
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 124,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 125,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 126,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 127,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 128,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 129,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 130,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 131,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 132,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 133,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 134,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 135,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 136,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 137,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 138,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk Throughput Usage",
				Summary: "Aggregated disk I/O rate, including the rates for all virtual machines running on the host during the collection interval",
			},
			Key: "throughput.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 139,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk Throughput Contention",
				Summary: "Average amount of time for an I/O operation to complete successfully",
			},
			Key: "throughput.contention",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 140,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk SCSI Reservation Conflicts",
				Summary: "Number of SCSI reservation conflicts for the LUN during the collection interval",
			},
			Key: "scsiReservationConflicts",
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 141,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk SCSI Reservation Conflicts %",
				Summary: "Number of SCSI reservation conflicts for the LUN as a percent of total commands during the collection interval",
			},
			Key: "scsiReservationCnflctsPct",
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
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 142,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 143,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 144,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 145,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 146,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 147,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 148,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 149,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 150,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Net Throughput Provisioned",
				Summary: "The maximum network bandwidth for the host",
			},
			Key: "throughput.provisioned",
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
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 151,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Net Throughput Usable",
				Summary: "The current available network bandwidth for the host",
			},
			Key: "throughput.usable",
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
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 152,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Net Throughput Usage",
				Summary: "The current network bandwidth usage for the host",
			},
			Key: "throughput.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 153,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Net Throughput Contention",
				Summary: "The aggregate network droppped packets for the host",
			},
			Key: "throughput.contention",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 154,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Packets Received and Transmitted per Second",
				Summary: "Average rate of packets received and transmitted per second",
			},
			Key: "throughput.packetsPerSec",
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
		RollupType:          "average",
		StatsType:           "rate",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 155,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 156,
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 157,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 158,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 159,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 160,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host Power Capacity Provisioned",
				Summary: "Current power usage as a percentage of maximum allowed power.",
			},
			Key: "capacity.usagePct",
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
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 161,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 162,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 163,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 164,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 165,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 166,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 167,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 168,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 169,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Throughput Contention",
				Summary: "Average amount of time for an I/O operation to complete successfully",
			},
			Key: "throughput.cont",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 170,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Outstanding I/Os",
				Summary: "The percent of I/Os that have been issued but have not yet completed",
			},
			Key: "OIOsPct",
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
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 171,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 172,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 173,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 174,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 175,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 176,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 177,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Disk Throughput Contention",
				Summary: "Average amount of time for an I/O operation to complete successfully",
			},
			Key: "throughput.cont",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 178,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 179,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 180,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 181,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 182,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 183,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 184,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 185,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 186,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 187,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "usage",
				Summary: "usage",
			},
			Key: "throughput.usage",
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
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 188,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "contention",
				Summary: "contention",
			},
			Key: "throughput.contention",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 189,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "busResets",
				Summary: "busResets",
			},
			Key: "busResets",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 190,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "commandsAborted",
				Summary: "commandsAborted",
			},
			Key: "commandsAborted",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 191,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 192,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Path Throughput Contention",
				Summary: "Average amount of time for an I/O operation to complete successfully",
			},
			Key: "throughput.cont",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 193,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 194,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Disk Throughput Usage",
				Summary: "Virtual disk I/O rate",
			},
			Key: "throughput.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 195,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Disk Number of Terminations",
				Summary: "Number of terminations to a virtual disk",
			},
			Key: "commandsAborted",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 196,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual Disk Number of Resets",
				Summary: "Number of resets to a virtual disk",
			},
			Key: "busResets",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 197,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Outstanding I/Os",
				Summary: "The number of I/Os that have been issued but have not yet completed",
			},
			Key: "outstandingIOs",
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
		StatsType:           "absolute",
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 198,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Number Queued",
				Summary: "The current number of I/Os that are waiting to be issued",
			},
			Key: "queued",
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
		StatsType:           "absolute",
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 199,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Queue Depth",
				Summary: "The maximum number of I/Os that can be outstanding at a given time",
			},
			Key: "queueDepth",
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
		StatsType:           "absolute",
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 200,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Queue Command Latency",
				Summary: "Average amount of time spent in the VMkernel queue, per SCSI command, during the collection interval",
			},
			Key: "queueLatency",
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 201,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Adapter Throughput Usage",
				Summary: "The storage adapter's I/O rate",
			},
			Key: "throughput.usag",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 202,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Path Bus Resets",
				Summary: "Number of SCSI-bus reset commands issued during the collection interval",
			},
			Key: "busResets",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 203,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Path Command Terminations",
				Summary: "Number of SCSI commands terminated during the collection interval",
			},
			Key: "commandsAborted",
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
		RollupType:          "summation",
		StatsType:           "delta",
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 204,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage Path Throughput Usage",
				Summary: "Storage path I/O rate",
			},
			Key: "throughput.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 205,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for VMs",
				Summary: "Average pNic I/O rate for VMs",
			},
			Key: "throughput.usage.vm",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 206,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for NFS",
				Summary: "Average pNic I/O rate for NFS",
			},
			Key: "throughput.usage.nfs",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 207,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for vMotion",
				Summary: "Average pNic I/O rate for vMotion",
			},
			Key: "throughput.usage.vmotion",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 208,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for FT",
				Summary: "Average pNic I/O rate for FT",
			},
			Key: "throughput.usage.ft",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 209,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for iSCSI",
				Summary: "Average pNic I/O rate for iSCSI",
			},
			Key: "throughput.usage.iscsi",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 210,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "pNic Throughput Usage for HBR",
				Summary: "Average pNic I/O rate for HBR",
			},
			Key: "throughput.usage.hbr",
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 211,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host Power Capacity Usable",
				Summary: "Current maximum allowed power usage.",
			},
			Key: "capacity.usable",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 212,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host Power Capacity Usage",
				Summary: "Current power usage",
			},
			Key: "capacity.usage",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 213,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Worst case allocation",
				Summary: "Amount of CPU resources allocated to the virtual machine or resource pool, based on the total cluster capacity and the resource configuration of the resource hierarchy",
			},
			Key: "cpuentitlement",
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 214,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Entitlement",
				Summary: "Memory allocation as calculated by the VMkernel scheduler based on current estimated demand and reservation, limit, and shares policies set for all virtual machines and resource pools in the host or cluster",
			},
			Key: "mementitlement",
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 215,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU fairness",
				Summary: "Fairness of distributed CPU resource allocation",
			},
			Key: "cpufairness",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cluster services",
				Summary: "Cluster services",
			},
			Key: "clusterServices",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 216,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Memory fairness",
				Summary: "Aggregate available memory resources of all the hosts within a cluster",
			},
			Key: "memfairness",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cluster services",
				Summary: "Cluster services",
			},
			Key: "clusterServices",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 217,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Packets Throughput Transmitted",
				Summary: "The rate of transmitted packets for this VDS",
			},
			Key: "throughput.pktsTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 218,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Multicast Packets Throughput Transmitted",
				Summary: "The rate of transmitted Multicast packets for this VDS",
			},
			Key: "throughput.pktsTxMulticast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 219,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Broadcast Packets Throughput Transmitted",
				Summary: "The rate of transmitted Broadcast packets for this VDS",
			},
			Key: "throughput.pktsTxBroadcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 220,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Packets Throughput Received",
				Summary: "The rate of received packets for this vDS",
			},
			Key: "throughput.pktsRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 221,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Multicast Packets Throughput Received",
				Summary: "The rate of received Multicast packets for this VDS",
			},
			Key: "throughput.pktsRxMulticast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 222,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Broadcast Packets Throughput Received",
				Summary: "The rate of received Broadcast packets for this VDS",
			},
			Key: "throughput.pktsRxBroadcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 223,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Dropped Transmitted Packets Throughput",
				Summary: "Count of dropped transmitted packets for this VDS",
			},
			Key: "throughput.droppedTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 224,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VDS Dropped Received Packets Throughput",
				Summary: "Count of dropped received packets for this VDS",
			},
			Key: "throughput.droppedRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 225,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Packets Throughput Transmitted",
				Summary: "The rate of transmitted packets for this DVPort",
			},
			Key: "throughput.vds.pktsTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 226,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Multicast Packets Throughput Transmitted",
				Summary: "The rate of transmitted multicast packets for this DVPort",
			},
			Key: "throughput.vds.pktsTxMcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 227,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Broadcast Packets Throughput Transmitted",
				Summary: "The rate of transmitted broadcast packets for this DVPort",
			},
			Key: "throughput.vds.pktsTxBcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 228,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Packets Throughput Received",
				Summary: "The rate of received packets for this DVPort",
			},
			Key: "throughput.vds.pktsRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 229,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Multicast Packets Throughput Received",
				Summary: "The rate of received multicast packets for this DVPort",
			},
			Key: "throughput.vds.pktsRxMcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 230,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort Broadcast Packets Throughput Received",
				Summary: "The rate of received broadcast packets for this DVPort",
			},
			Key: "throughput.vds.pktsRxBcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 231,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort dropped transmitted packets throughput",
				Summary: "Count of dropped transmitted packets for this DVPort",
			},
			Key: "throughput.vds.droppedTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 232,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "DVPort dropped received packets throughput",
				Summary: "Count of dropped received packets for this DVPort",
			},
			Key: "throughput.vds.droppedRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 233,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG Packets Throughput Transmitted",
				Summary: "The rate of transmitted packets for this LAG",
			},
			Key: "throughput.vds.lagTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 234,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG Multicast Packets Throughput Transmitted",
				Summary: "The rate of transmitted Multicast packets for this LAG",
			},
			Key: "throughput.vds.lagTxMcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 235,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG Broadcast Packets Throughput Transmitted",
				Summary: "The rate of transmitted Broadcast packets for this LAG",
			},
			Key: "throughput.vds.lagTxBcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 236,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG packets Throughput received",
				Summary: "The rate of received packets for this LAG",
			},
			Key: "throughput.vds.lagRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 237,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG multicast packets throughput received",
				Summary: "The rate of received multicast packets for this LAG",
			},
			Key: "throughput.vds.lagRxMcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 238,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG Broadcast packets Throughput received",
				Summary: "The rate of received Broadcast packets for this LAG",
			},
			Key: "throughput.vds.lagRxBcast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 239,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG dropped transmitted packets throughput",
				Summary: "Count of dropped transmitted packets for this LAG",
			},
			Key: "throughput.vds.lagDropTx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 240,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "LAG dropped received packets throughput",
				Summary: "Count of dropped received packets for this LAG",
			},
			Key: "throughput.vds.lagDropRx",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 241,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network packets throughput transmitted",
				Summary: "The rate of transmitted packets for this network",
			},
			Key: "throughput.vds.txTotal",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 242,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network non-unicast packets throughput transmitted",
				Summary: "The rate of transmitted non-unicast packets for this network",
			},
			Key: "throughput.vds.txNoUnicast",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 243,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network cross-router packets throughput transmitted",
				Summary: "The rate of transmitted cross-router packets for this network",
			},
			Key: "throughput.vds.txCrsRouter",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 244,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped transmitted packets throughput",
				Summary: "Count of dropped transmitted packets for this network",
			},
			Key: "throughput.vds.txDrop",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 245,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network packets throughput received",
				Summary: "The rate of received packets for this network",
			},
			Key: "throughput.vds.rxTotal",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 246,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped received packets due to destination IP error throughput",
				Summary: "Count of dropped received packets with destination IP error for this network",
			},
			Key: "throughput.vds.rxDestErr",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 247,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped received packets throughput",
				Summary: "Count of dropped received packets for this network",
			},
			Key: "throughput.vds.rxDrop",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 248,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to match mapping entry for a unicast MAC throughput",
				Summary: "Count of transmitted packets that cannot find matched mapping entry for this network",
			},
			Key: "throughput.vds.macFlood",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 249,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to allocate a new mapping entry during translation phase",
				Summary: "Count of transmitted packets that failed to acquire new mapping entry during translation phase for this network",
			},
			Key: "throughput.vds.macLKUPFull",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 250,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to allocate a new mapping entry during learning phase",
				Summary: "Count of transmitted packets that failed to acquire new mapping entry during learning phase for this network",
			},
			Key: "throughput.vds.macUPDTFull",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 251,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Nework Found Matched ARP Entry Throughput",
				Summary: "Count of transmitted packets that found matched ARP entry for this network",
			},
			Key: "throughput.vds.arpFound",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 252,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Found Matched ARP Entry Marked as Unknown Throughput",
				Summary: "Count of transmitted packets whose matched arp entry is marked as unknown for this network",
			},
			Key: "throughput.vds.arpUnknown",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 253,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Failed to Allocate ARP Entry During Translation Phase Throughput",
				Summary: "Count of transmitted packets that failed to acquire new ARP entry during translation phase for this network",
			},
			Key: "throughput.vds.arpLKUPFull",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 254,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network found the same ARP requests have been sent into queue throughput",
				Summary: "Count of transmitted packets whose ARP requests have already been sent into queue for this network",
			},
			Key: "throughput.vds.arpWait",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 255,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Found ARP Queries Have Been Expired Throughput",
				Summary: "Count of arp queries that have been expired for this network",
			},
			Key: "throughput.vds.arpTimeout",
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
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 256,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM power on count",
				Summary: "Number of virtual machine power on operations",
			},
			Key: "numPoweron",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 257,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM power off count",
				Summary: "Number of virtual machine power off operations",
			},
			Key: "numPoweroff",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 258,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM suspend count",
				Summary: "Number of virtual machine suspend operations",
			},
			Key: "numSuspend",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 259,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM reset count",
				Summary: "Number of virtual machine reset operations",
			},
			Key: "numReset",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 260,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM guest reboot count",
				Summary: "Number of virtual machine guest reboot operations",
			},
			Key: "numRebootGuest",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 261,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM standby guest count",
				Summary: "Number of virtual machine standby guest operations",
			},
			Key: "numStandbyGuest",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 262,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM guest shutdown count",
				Summary: "Number of virtual machine guest shutdown operations",
			},
			Key: "numShutdownGuest",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 263,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM create count",
				Summary: "Number of virtual machine create operations",
			},
			Key: "numCreate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 264,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM delete count",
				Summary: "Number of virtual machine delete operations",
			},
			Key: "numDestroy",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 265,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM register count",
				Summary: "Number of virtual machine register operations",
			},
			Key: "numRegister",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 266,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM unregister count",
				Summary: "Number of virtual machine unregister operations",
			},
			Key: "numUnregister",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 267,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM reconfigure count",
				Summary: "Number of virtual machine reconfigure operations",
			},
			Key: "numReconfigure",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 268,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM clone count",
				Summary: "Number of virtual machine clone operations",
			},
			Key: "numClone",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 269,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM template deploy count",
				Summary: "Number of virtual machine template deploy operations",
			},
			Key: "numDeploy",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 270,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM host change count (non-powered-on VMs)",
				Summary: "Number of host change operations for powered-off and suspended VMs",
			},
			Key: "numChangeHost",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 271,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM datastore change count (non-powered-on VMs)",
				Summary: "Number of datastore change operations for powered-off and suspended virtual machines",
			},
			Key: "numChangeDS",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 272,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM host and datastore change count (non-powered-on VMs)",
				Summary: "Number of host and datastore change operations for powered-off and suspended virtual machines",
			},
			Key: "numChangeHostDS",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 273,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vMotion count",
				Summary: "Number of migrations with vMotion (host change operations for powered-on VMs)",
			},
			Key: "numVMotion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 274,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Storage vMotion count",
				Summary: "Number of migrations with Storage vMotion (datastore change operations for powered-on VMs)",
			},
			Key: "numSVMotion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 275,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VM host and datastore change count (powered-on VMs)",
				Summary: "Number of host and datastore change operations for powered-on and suspended virtual machines",
			},
			Key: "numXVMotion",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual machine operations",
				Summary: "Virtual machine operations",
			},
			Key: "vmop",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 276,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Effective CPU resources",
				Summary: "Total available CPU resources of all hosts within a cluster",
			},
			Key: "effectivecpu",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cluster services",
				Summary: "Cluster services",
			},
			Key: "clusterServices",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 277,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Effective memory resources",
				Summary: "Total amount of machine memory of all hosts in the cluster that is available for use for virtual machine memory and overhead memory",
			},
			Key: "effectivemem",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cluster services",
				Summary: "Cluster services",
			},
			Key: "clusterServices",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 278,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Total",
				Summary: "Total amount of CPU resources of all hosts in the cluster",
			},
			Key: "totalmhz",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 279,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Total",
				Summary: "Total amount of host physical memory of all hosts in the cluster that is available for virtual machine memory (physical memory for use by the guest OS) and virtual machine overhead memory",
			},
			Key: "totalmb",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 280,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Current failover level",
				Summary: "vSphere HA number of failures that can be tolerated",
			},
			Key: "failover",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Cluster services",
				Summary: "Cluster services",
			},
			Key: "clusterServices",
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 281,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Space actually used",
				Summary: "Amount of space actually used by the virtual machine or the datastore",
			},
			Key: "used",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 282,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Space potentially used",
				Summary: "Amount of storage set aside for use by a datastore or a virtual machine",
			},
			Key: "provisioned",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 283,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Capacity",
				Summary: "Configured size of the datastore",
			},
			Key: "capacity",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 284,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Space not shared",
				Summary: "Amount of space associated exclusively with a virtual machine",
			},
			Key: "unshared",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 285,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Overhead due to delta disk backings",
				Summary: "Storage overhead of a virtual machine or a datastore due to delta disk backings",
			},
			Key: "deltaused",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "latest",
		StatsType:           "absolute",
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 286,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "provisioned",
				Summary: "provisioned",
			},
			Key: "capacity.provisioned",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 287,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "usage",
				Summary: "usage",
			},
			Key: "capacity.usage",
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
				Label:   "KB",
				Summary: "Kilobyte",
			},
			Key: "kiloBytes",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 288,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "contention",
				Summary: "contention",
			},
			Key: "capacity.contention",
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
				Label:   "%",
				Summary: "Percentage",
			},
			Key: "percent",
		},
		RollupType:          "average",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 289,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation latency",
				Summary: "The latency of an activation operation in vCenter Server",
			},
			Key: "activationlatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 290,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation latency",
				Summary: "The latency of an activation operation in vCenter Server",
			},
			Key: "activationlatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 291,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation latency",
				Summary: "The latency of an activation operation in vCenter Server",
			},
			Key: "activationlatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 292,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation count",
				Summary: "Activation operations in vCenter Server",
			},
			Key: "activationstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 293,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation count",
				Summary: "Activation operations in vCenter Server",
			},
			Key: "activationstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 294,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Activation count",
				Summary: "Activation operations in vCenter Server",
			},
			Key: "activationstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 295,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "buffersz",
				Summary: "buffersz",
			},
			Key: "buffersz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 296,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "cachesz",
				Summary: "cachesz",
			},
			Key: "cachesz",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 297,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Context switch rate",
				Summary: "Number of context switches per second on the system where vCenter Server is running",
			},
			Key: "ctxswitchesrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 298,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "diskreadsectorrate",
				Summary: "diskreadsectorrate",
			},
			Key: "diskreadsectorrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 299,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk read rate",
				Summary: "Number of disk reads per second on the system where vCenter Server is running",
			},
			Key: "diskreadsrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 300,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "diskwritesectorrate",
				Summary: "diskwritesectorrate",
			},
			Key: "diskwritesectorrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 301,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Disk write rate",
				Summary: "Number of disk writes per second on the system where vCenter Server is running",
			},
			Key: "diskwritesrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 302,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync latency",
				Summary: "The latency of a host sync operation in vCenter Server",
			},
			Key: "hostsynclatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 303,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync latency",
				Summary: "The latency of a host sync operation in vCenter Server",
			},
			Key: "hostsynclatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 304,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync latency",
				Summary: "The latency of a host sync operation in vCenter Server",
			},
			Key: "hostsynclatencystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "ms",
				Summary: "Millisecond",
			},
			Key: "millisecond",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 305,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync count",
				Summary: "The number of host sync operations in vCenter Server",
			},
			Key: "hostsyncstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 306,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync count",
				Summary: "The number of host sync operations in vCenter Server",
			},
			Key: "hostsyncstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 307,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Host sync count",
				Summary: "The number of host sync operations in vCenter Server",
			},
			Key: "hostsyncstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 308,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Inventory statistics",
				Summary: "vCenter Server inventory statistics",
			},
			Key: "inventorystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 309,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Inventory statistics",
				Summary: "vCenter Server inventory statistics",
			},
			Key: "inventorystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 310,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Inventory statistics",
				Summary: "vCenter Server inventory statistics",
			},
			Key: "inventorystats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 311,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Locking statistics",
				Summary: "vCenter Server locking statistics",
			},
			Key: "lockstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 312,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Locking statistics",
				Summary: "vCenter Server locking statistics",
			},
			Key: "lockstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 313,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Locking statistics",
				Summary: "vCenter Server locking statistics",
			},
			Key: "lockstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 314,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server LRO statistics",
				Summary: "vCenter Server LRO statistics",
			},
			Key: "lrostats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 315,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server LRO statistics",
				Summary: "vCenter Server LRO statistics",
			},
			Key: "lrostats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 316,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server LRO statistics",
				Summary: "vCenter Server LRO statistics",
			},
			Key: "lrostats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 317,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Miscellaneous",
				Summary: "Miscellaneous statistics",
			},
			Key: "miscstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 318,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Miscellaneous",
				Summary: "Miscellaneous statistics",
			},
			Key: "miscstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 319,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Miscellaneous",
				Summary: "Miscellaneous statistics",
			},
			Key: "miscstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 320,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Managed object reference statistics",
				Summary: "Managed object reference counts in vCenter Server",
			},
			Key: "morefregstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 321,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Managed object reference statistics",
				Summary: "Managed object reference counts in vCenter Server",
			},
			Key: "morefregstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 322,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Managed object reference statistics",
				Summary: "Managed object reference counts in vCenter Server",
			},
			Key: "morefregstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 323,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Received packet rate",
				Summary: "Rate of the number of total packets received per second on the system where vCenter Server is running",
			},
			Key: "packetrecvrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 324,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Sent packet rate",
				Summary: "Number of total packets sent per second on the system where vCenter Server is running",
			},
			Key: "packetsentrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 325,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU system",
				Summary: "Total system CPU used on the system where vCenter Server in running",
			},
			Key: "systemcpuusage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 326,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Page fault rate",
				Summary: "Number of page faults per second on the system where vCenter Server is running",
			},
			Key: "pagefaultrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 327,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Physical memory",
				Summary: "Physical memory used by vCenter",
			},
			Key: "physicalmemusage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 328,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU privileged",
				Summary: "CPU used by vCenter Server in privileged mode",
			},
			Key: "priviledgedcpuusage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 329,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Scoreboard statistics",
				Summary: "Object counts in vCenter Server",
			},
			Key: "scoreboard",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 330,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Scoreboard statistics",
				Summary: "Object counts in vCenter Server",
			},
			Key: "scoreboard",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 331,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Scoreboard statistics",
				Summary: "Object counts in vCenter Server",
			},
			Key: "scoreboard",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 332,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Session statistics",
				Summary: "The statistics of client sessions connected to vCenter Server",
			},
			Key: "sessionstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 333,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Session statistics",
				Summary: "The statistics of client sessions connected to vCenter Server",
			},
			Key: "sessionstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 334,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Session statistics",
				Summary: "The statistics of client sessions connected to vCenter Server",
			},
			Key: "sessionstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 335,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System call rate",
				Summary: "Number of systems calls made per second on the system where vCenter Server is running",
			},
			Key: "syscallsrate",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 336,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System statistics",
				Summary: "The statistics of vCenter Server as a running system such as thread statistics and heap statistics",
			},
			Key: "systemstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 337,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System statistics",
				Summary: "The statistics of vCenter Server as a running system such as thread statistics and heap statistics",
			},
			Key: "systemstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 338,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "System statistics",
				Summary: "The statistics of vCenter Server as a running system such as thread statistics and heap statistics",
			},
			Key: "systemstats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 339,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "CPU user",
				Summary: "CPU used by vCenter Server in user mode",
			},
			Key: "usercpuusage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 340,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server service statistics",
				Summary: "vCenter service statistics such as events, alarms, and tasks",
			},
			Key: "vcservicestats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "maximum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 341,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server service statistics",
				Summary: "vCenter service statistics such as events, alarms, and tasks",
			},
			Key: "vcservicestats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "minimum",
		StatsType:           "absolute",
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 342,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter Server service statistics",
				Summary: "vCenter service statistics such as events, alarms, and tasks",
			},
			Key: "vcservicestats",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter debugging information",
				Summary: "vCenter debugging information",
			},
			Key: "vcDebugInfo",
		},
		UnitInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "num",
				Summary: "Number",
			},
			Key: "number",
		},
		RollupType:          "summation",
		StatsType:           "absolute",
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 343,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "Virtual memory",
				Summary: "Virtual memory used by vCenter Server",
			},
			Key: "virtualmemusage",
		},
		GroupInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "vCenter resource usage information",
				Summary: "vCenter resource usage information",
			},
			Key: "vcResources",
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
		Level:               1,
		PerDeviceLevel:      1,
		AssociatedCounterId: nil,
	},
	{
		Key: 344,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 345,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 346,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 347,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 348,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 349,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 350,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 351,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 352,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 353,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 354,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 355,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 356,
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
		Level:               2,
		PerDeviceLevel:      2,
		AssociatedCounterId: nil,
	},
	{
		Key: 357,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 358,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 359,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 360,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network packets throughput transmitted",
				Summary: "The rate of transmitted packets for this network",
			},
			Key: "throughput.vds.txTotal",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 361,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network non-unicast packets throughput transmitted",
				Summary: "The rate of transmitted non-unicast packets for this network",
			},
			Key: "throughput.vds.txNoUnicast",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 362,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network cross-router packets throughput transmitted",
				Summary: "The rate of transmitted cross-router packets for this network",
			},
			Key: "throughput.vds.txCrsRouter",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 363,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped transmitted packets throughput",
				Summary: "Count of dropped transmitted packets for this network",
			},
			Key: "throughput.vds.txDrop",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 364,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network packets throughput received",
				Summary: "The rate of received packets for this network",
			},
			Key: "throughput.vds.rxTotal",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 365,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped received packets due to destination IP error throughput",
				Summary: "Count of dropped received packets with destination IP error for this network",
			},
			Key: "throughput.vds.rxDestErr",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 366,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network dropped received packets throughput",
				Summary: "Count of dropped received packets for this network",
			},
			Key: "throughput.vds.rxDrop",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 367,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to match mapping entry for a unicast MAC throughput",
				Summary: "Count of transmitted packets that cannot find matched mapping entry for this network",
			},
			Key: "throughput.vds.macFlood",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 368,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to allocate a new mapping entry during translation phase",
				Summary: "Count of transmitted packets that failed to acquire new mapping entry during translation phase for this network",
			},
			Key: "throughput.vds.macLKUPFull",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 369,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network failed to allocate a new mapping entry during learning phase",
				Summary: "Count of transmitted packets that failed to acquire new mapping entry during learning phase for this network",
			},
			Key: "throughput.vds.macUPDTFull",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 370,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Nework Found Matched ARP Entry Throughput",
				Summary: "Count of transmitted packets that found matched ARP entry for this network",
			},
			Key: "throughput.vds.arpFound",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 371,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Found Matched ARP Entry Marked as Unknown Throughput",
				Summary: "Count of transmitted packets whose matched arp entry is marked as unknown for this network",
			},
			Key: "throughput.vds.arpUnknown",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 372,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Failed to Allocate ARP Entry During Translation Phase Throughput",
				Summary: "Count of transmitted packets that failed to acquire new ARP entry during translation phase for this network",
			},
			Key: "throughput.vds.arpLKUPFull",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 373,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN network found the same ARP requests have been sent into queue throughput",
				Summary: "Count of transmitted packets whose ARP requests have already been sent into queue for this network",
			},
			Key: "throughput.vds.arpWait",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 374,
		NameInfo: &types.ElementDescription{
			Description: types.Description{
				Label:   "VXLAN Network Found ARP Queries Have Been Expired Throughput",
				Summary: "Count of arp queries that have been expired for this network",
			},
			Key: "throughput.vds.arpTimeout",
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 386,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 387,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 388,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 389,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 390,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 391,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 392,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 393,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 394,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 395,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 396,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 397,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 398,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 399,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 400,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 401,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 402,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 403,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 404,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 405,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 406,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 407,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 408,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 409,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 410,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 411,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 412,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 413,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 414,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 415,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 416,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 417,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 418,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 419,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 420,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 421,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 422,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 423,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 424,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 425,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 426,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 427,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 428,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 429,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 430,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 431,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 432,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 433,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 434,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 435,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 436,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 437,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 438,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 439,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 440,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 441,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 442,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 443,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 444,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 445,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 446,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 447,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 448,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 449,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 450,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 451,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 452,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 453,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 454,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 455,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 456,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 457,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 458,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 459,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 460,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 461,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 462,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 463,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 464,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 465,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 466,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 467,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 468,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 469,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 470,
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
		Level:               2,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 471,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 472,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 473,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 474,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 475,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 476,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 477,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 478,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 479,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 480,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 481,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 482,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 483,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 484,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 485,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 486,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 487,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 488,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 489,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 490,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 491,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 492,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 493,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 494,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 495,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 496,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 497,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 498,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 499,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 500,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 501,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 502,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 503,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 504,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 505,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 506,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 507,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 508,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 509,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 510,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 511,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 512,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 513,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 514,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 515,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 516,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 517,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 518,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 519,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 520,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 521,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 522,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 523,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 524,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 525,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 526,
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
		Level:               3,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 527,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 528,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 529,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 530,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 531,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 532,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 533,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 534,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 535,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 536,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 537,
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
		Level:               1,
		PerDeviceLevel:      3,
		AssociatedCounterId: nil,
	},
	{
		Key: 538,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 539,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 540,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 541,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 542,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 543,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 544,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 545,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 546,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 547,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 548,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 549,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 550,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 551,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 552,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 553,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 554,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 555,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 556,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 557,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 558,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 559,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 560,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 561,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 562,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 563,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 564,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 565,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 566,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 567,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 568,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 569,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 570,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
	{
		Key: 571,
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
		Level:               4,
		PerDeviceLevel:      4,
		AssociatedCounterId: nil,
	},
}

var VmMetrics = []types.PerfMetricId{
	{
		CounterId: 12,
		Instance:  "$cpu",
	},
	{
		CounterId: 401,
		Instance:  "$cpu",
	},
	{
		CounterId: 2,
		Instance:  "",
	},
	{
		CounterId: 14,
		Instance:  "",
	},
	{
		CounterId: 10,
		Instance:  "",
	},
	{
		CounterId: 401,
		Instance:  "",
	},
	{
		CounterId: 402,
		Instance:  "$cpu",
	},
	{
		CounterId: 396,
		Instance:  "",
	},
	{
		CounterId: 13,
		Instance:  "",
	},
	{
		CounterId: 11,
		Instance:  "$cpu",
	},
	{
		CounterId: 386,
		Instance:  "$cpu",
	},
	{
		CounterId: 399,
		Instance:  "$cpu",
	},
	{
		CounterId: 397,
		Instance:  "",
	},
	{
		CounterId: 6,
		Instance:  "$cpu",
	},
	{
		CounterId: 404,
		Instance:  "$cpu",
	},
	{
		CounterId: 386,
		Instance:  "",
	},
	{
		CounterId: 14,
		Instance:  "$cpu",
	},
	{
		CounterId: 11,
		Instance:  "",
	},
	{
		CounterId: 400,
		Instance:  "$cpu",
	},
	{
		CounterId: 6,
		Instance:  "",
	},
	{
		CounterId: 399,
		Instance:  "",
	},
	{
		CounterId: 403,
		Instance:  "",
	},
	{
		CounterId: 404,
		Instance:  "",
	},
	{
		CounterId: 398,
		Instance:  "",
	},
	{
		CounterId: 13,
		Instance:  "$cpu",
	},
	{
		CounterId: 400,
		Instance:  "",
	},
	{
		CounterId: 402,
		Instance:  "",
	},
	{
		CounterId: 12,
		Instance:  "",
	},
	{
		CounterId: 184,
		Instance:  "",
	},
	{
		CounterId: 180,
		Instance:  "$physDisk",
	},
	{
		CounterId: 181,
		Instance:  "$physDisk",
	},
	{
		CounterId: 178,
		Instance:  "$physDisk",
	},
	{
		CounterId: 182,
		Instance:  "$physDisk",
	},
	{
		CounterId: 179,
		Instance:  "$physDisk",
	},
	{
		CounterId: 183,
		Instance:  "$physDisk",
	},
	{
		CounterId: 133,
		Instance:  "",
	},
	{
		CounterId: 37,
		Instance:  "",
	},
	{
		CounterId: 74,
		Instance:  "",
	},
	{
		CounterId: 426,
		Instance:  "",
	},
	{
		CounterId: 70,
		Instance:  "",
	},
	{
		CounterId: 107,
		Instance:  "",
	},
	{
		CounterId: 422,
		Instance:  "",
	},
	{
		CounterId: 105,
		Instance:  "",
	},
	{
		CounterId: 85,
		Instance:  "",
	},
	{
		CounterId: 428,
		Instance:  "",
	},
	{
		CounterId: 418,
		Instance:  "",
	},
	{
		CounterId: 102,
		Instance:  "",
	},
	{
		CounterId: 33,
		Instance:  "",
	},
	{
		CounterId: 427,
		Instance:  "",
	},
	{
		CounterId: 94,
		Instance:  "",
	},
	{
		CounterId: 29,
		Instance:  "",
	},
	{
		CounterId: 420,
		Instance:  "",
	},
	{
		CounterId: 417,
		Instance:  "",
	},
	{
		CounterId: 98,
		Instance:  "",
	},
	{
		CounterId: 423,
		Instance:  "",
	},
	{
		CounterId: 106,
		Instance:  "",
	},
	{
		CounterId: 86,
		Instance:  "",
	},
	{
		CounterId: 41,
		Instance:  "",
	},
	{
		CounterId: 421,
		Instance:  "",
	},
	{
		CounterId: 429,
		Instance:  "",
	},
	{
		CounterId: 406,
		Instance:  "",
	},
	{
		CounterId: 90,
		Instance:  "",
	},
	{
		CounterId: 24,
		Instance:  "",
	},
	{
		CounterId: 410,
		Instance:  "",
	},
	{
		CounterId: 149,
		Instance:  "vmnic1",
	},
	{
		CounterId: 466,
		Instance:  "4000",
	},
	{
		CounterId: 146,
		Instance:  "",
	},
	{
		CounterId: 461,
		Instance:  "",
	},
	{
		CounterId: 148,
		Instance:  "vmnic1",
	},
	{
		CounterId: 462,
		Instance:  "vmnic0",
	},
	{
		CounterId: 143,
		Instance:  "vmnic0",
	},
	{
		CounterId: 463,
		Instance:  "vmnic1",
	},
	{
		CounterId: 147,
		Instance:  "",
	},
	{
		CounterId: 463,
		Instance:  "4000",
	},
	{
		CounterId: 462,
		Instance:  "vmnic1",
	},
	{
		CounterId: 462,
		Instance:  "4000",
	},
	{
		CounterId: 461,
		Instance:  "4000",
	},
	{
		CounterId: 146,
		Instance:  "vmnic0",
	},
	{
		CounterId: 465,
		Instance:  "4000",
	},
	{
		CounterId: 460,
		Instance:  "4000",
	},
	{
		CounterId: 149,
		Instance:  "4000",
	},
	{
		CounterId: 148,
		Instance:  "4000",
	},
	{
		CounterId: 462,
		Instance:  "",
	},
	{
		CounterId: 149,
		Instance:  "vmnic0",
	},
	{
		CounterId: 143,
		Instance:  "4000",
	},
	{
		CounterId: 463,
		Instance:  "",
	},
	{
		CounterId: 147,
		Instance:  "vmnic1",
	},
	{
		CounterId: 466,
		Instance:  "",
	},
	{
		CounterId: 472,
		Instance:  "4000",
	},
	{
		CounterId: 143,
		Instance:  "",
	},
	{
		CounterId: 146,
		Instance:  "vmnic1",
	},
	{
		CounterId: 146,
		Instance:  "4000",
	},
	{
		CounterId: 472,
		Instance:  "",
	},
	{
		CounterId: 471,
		Instance:  "",
	},
	{
		CounterId: 460,
		Instance:  "",
	},
	{
		CounterId: 147,
		Instance:  "4000",
	},
	{
		CounterId: 471,
		Instance:  "4000",
	},
	{
		CounterId: 148,
		Instance:  "",
	},
	{
		CounterId: 147,
		Instance:  "vmnic0",
	},
	{
		CounterId: 465,
		Instance:  "",
	},
	{
		CounterId: 464,
		Instance:  "4000",
	},
	{
		CounterId: 464,
		Instance:  "",
	},
	{
		CounterId: 148,
		Instance:  "vmnic0",
	},
	{
		CounterId: 463,
		Instance:  "vmnic0",
	},
	{
		CounterId: 467,
		Instance:  "",
	},
	{
		CounterId: 143,
		Instance:  "vmnic1",
	},
	{
		CounterId: 149,
		Instance:  "",
	},
	{
		CounterId: 467,
		Instance:  "4000",
	},
	{
		CounterId: 159,
		Instance:  "",
	},
	{
		CounterId: 157,
		Instance:  "",
	},
	{
		CounterId: 504,
		Instance:  "",
	},
	{
		CounterId: 507,
		Instance:  "",
	},
	{
		CounterId: 513,
		Instance:  "",
	},
	{
		CounterId: 348,
		Instance:  "",
	},
	{
		CounterId: 505,
		Instance:  "",
	},
	{
		CounterId: 514,
		Instance:  "",
	},
	{
		CounterId: 506,
		Instance:  "",
	},
	{
		CounterId: 512,
		Instance:  "",
	},
	{
		CounterId: 508,
		Instance:  "",
	},
	{
		CounterId: 515,
		Instance:  "",
	},
	{
		CounterId: 509,
		Instance:  "",
	},
	{
		CounterId: 501,
		Instance:  "",
	},
	{
		CounterId: 516,
		Instance:  "",
	},
	{
		CounterId: 503,
		Instance:  "",
	},
	{
		CounterId: 511,
		Instance:  "",
	},
	{
		CounterId: 510,
		Instance:  "",
	},
	{
		CounterId: 502,
		Instance:  "",
	},
	{
		CounterId: 155,
		Instance:  "",
	},
	{
		CounterId: 473,
		Instance:  "",
	},
	{
		CounterId: 498,
		Instance:  "",
	},
	{
		CounterId: 174,
		Instance:  "",
	},
	{
		CounterId: 173,
		Instance:  "",
	},
}

// ************************* Host metrics ************************************

var HostMetrics = []types.PerfMetricId{
	{
		CounterId: 386,
		Instance:  "",
	},
	{
		CounterId: 395,
		Instance:  "",
	},
	{
		CounterId: 14,
		Instance:  "",
	},
	{
		CounterId: 399,
		Instance:  "",
	},
	{
		CounterId: 392,
		Instance:  "",
	},
	{
		CounterId: 392,
		Instance:  "$cpu",
	},
	{
		CounterId: 11,
		Instance:  "",
	},
	{
		CounterId: 398,
		Instance:  "",
	},
	{
		CounterId: 388,
		Instance:  "",
	},
	{
		CounterId: 388,
		Instance:  "$cpu",
	},
	{
		CounterId: 13,
		Instance:  "",
	},
	{
		CounterId: 396,
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
		CounterId: 2,
		Instance:  "",
	},
	{
		CounterId: 14,
		Instance:  "$cpu",
	},
	{
		CounterId: 404,
		Instance:  "",
	},
	{
		CounterId: 6,
		Instance:  "",
	},
	{
		CounterId: 2,
		Instance:  "$cpu",
	},
	{
		CounterId: 13,
		Instance:  "$cpu",
	},
	{
		CounterId: 185,
		Instance:  "d10c389e-c75b7dc4",
	},
	{
		CounterId: 179,
		Instance:  "$physDisk",
	},
	{
		CounterId: 178,
		Instance:  "$physDisk",
	},
	{
		CounterId: 358,
		Instance:  "$physDisk",
	},
	{
		CounterId: 537,
		Instance:  "$physDisk",
	},
	{
		CounterId: 354,
		Instance:  "$physDisk",
	},
	{
		CounterId: 191,
		Instance:  "$physDisk",
	},
	{
		CounterId: 352,
		Instance:  "$physDisk",
	},
	{
		CounterId: 359,
		Instance:  "$physDisk",
	},
	{
		CounterId: 184,
		Instance:  "",
	},
	{
		CounterId: 186,
		Instance:  "$physDisk",
	},
	{
		CounterId: 351,
		Instance:  "$physDisk",
	},
	{
		CounterId: 180,
		Instance:  "$physDisk",
	},
	{
		CounterId: 353,
		Instance:  "$physDisk",
	},
	{
		CounterId: 356,
		Instance:  "$physDisk",
	},
	{
		CounterId: 355,
		Instance:  "$physDisk",
	},
	{
		CounterId: 350,
		Instance:  "$physDisk",
	},
	{
		CounterId: 349,
		Instance:  "$physDisk",
	},
	{
		CounterId: 182,
		Instance:  "$physDisk",
	},
	{
		CounterId: 357,
		Instance:  "$physDisk",
	},
	{
		CounterId: 181,
		Instance:  "$physDisk",
	},
	{
		CounterId: 185,
		Instance:  "$physDisk",
	},
	{
		CounterId: 183,
		Instance:  "$physDisk",
	},

	{
		CounterId: 455,
		Instance:  "$physDisk",
	},
	{
		CounterId: 133,
		Instance:  "",
	},
	{
		CounterId: 456,
		Instance:  "$physDisk",
	},
	{
		CounterId: 457,
		Instance:  "$physDisk",
	},
	{
		CounterId: 129,
		Instance:  "$physDisk",
	},
	{
		CounterId: 448,
		Instance:  "$physDisk",
	},
	{
		CounterId: 130,
		Instance:  "",
	},
	{
		CounterId: 447,
		Instance:  "$physDisk",
	},
	{
		CounterId: 458,
		Instance:  "$physDisk",
	},
	{
		CounterId: 131,
		Instance:  "$physDisk",
	},
	{
		CounterId: 134,
		Instance:  "$physDisk",
	},
	{
		CounterId: 446,
		Instance:  "$physDisk",
	},
	{
		CounterId: 450,
		Instance:  "$physDisk",
	},
	{
		CounterId: 451,
		Instance:  "$physDisk",
	},
	{
		CounterId: 453,
		Instance:  "$physDisk",
	},
	{
		CounterId: 452,
		Instance:  "$physDisk",
	},
	{
		CounterId: 454,
		Instance:  "$physDisk",
	},
	{
		CounterId: 128,
		Instance:  "$physDisk",
	},
	{
		CounterId: 132,
		Instance:  "$physDisk",
	},
	{
		CounterId: 459,
		Instance:  "$physDisk",
	},
	{
		CounterId: 130,
		Instance:  "$physDisk",
	},
	{
		CounterId: 125,
		Instance:  "",
	},
	{
		CounterId: 131,
		Instance:  "",
	},
	{
		CounterId: 449,
		Instance:  "$physDisk",
	},
	{
		CounterId: 135,
		Instance:  "$physDisk",
	},
	{
		CounterId: 136,
		Instance:  "$physDisk",
	},
	{
		CounterId: 137,
		Instance:  "$physDisk",
	},
	{
		CounterId: 538,
		Instance:  "",
	},
	{
		CounterId: 540,
		Instance:  "",
	},
	{
		CounterId: 539,
		Instance:  "",
	},
	{
		CounterId: 65,
		Instance:  "",
	},
	{
		CounterId: 27,
		Instance:  "",
	},
	{
		CounterId: 419,
		Instance:  "",
	},
	{
		CounterId: 443,
		Instance:  "",
	},
	{
		CounterId: 437,
		Instance:  "",
	},
	{
		CounterId: 24,
		Instance:  "",
	},
	{
		CounterId: 68,
		Instance:  "",
	},
	{
		CounterId: 422,
		Instance:  "",
	},
	{
		CounterId: 106,
		Instance:  "",
	},
	{
		CounterId: 410,
		Instance:  "",
	},
	{
		CounterId: 33,
		Instance:  "",
	},
	{
		CounterId: 105,
		Instance:  "",
	},
	{
		CounterId: 107,
		Instance:  "",
	},
	{
		CounterId: 61,
		Instance:  "",
	},
	{
		CounterId: 445,
		Instance:  "",
	},
	{
		CounterId: 417,
		Instance:  "",
	},
	{
		CounterId: 406,
		Instance:  "",
	},
	{
		CounterId: 444,
		Instance:  "",
	},
	{
		CounterId: 427,
		Instance:  "",
	},
	{
		CounterId: 85,
		Instance:  "",
	},
	{
		CounterId: 424,
		Instance:  "",
	},
	{
		CounterId: 49,
		Instance:  "",
	},
	{
		CounterId: 414,
		Instance:  "",
	},
	{
		CounterId: 98,
		Instance:  "",
	},
	{
		CounterId: 29,
		Instance:  "",
	},
	{
		CounterId: 57,
		Instance:  "",
	},
	{
		CounterId: 441,
		Instance:  "",
	},
	{
		CounterId: 41,
		Instance:  "",
	},
	{
		CounterId: 86,
		Instance:  "",
	},
	{
		CounterId: 433,
		Instance:  "",
	},
	{
		CounterId: 45,
		Instance:  "",
	},
	{
		CounterId: 426,
		Instance:  "",
	},
	{
		CounterId: 429,
		Instance:  "",
	},
	{
		CounterId: 440,
		Instance:  "",
	},
	{
		CounterId: 102,
		Instance:  "",
	},
	{
		CounterId: 90,
		Instance:  "",
	},
	{
		CounterId: 37,
		Instance:  "",
	},
	{
		CounterId: 442,
		Instance:  "",
	},
	{
		CounterId: 469,
		Instance:  "vmnic0",
	},
	{
		CounterId: 460,
		Instance:  "",
	},
	{
		CounterId: 463,
		Instance:  "",
	},
	{
		CounterId: 143,
		Instance:  "",
	},
	{
		CounterId: 465,
		Instance:  "",
	},
	{
		CounterId: 461,
		Instance:  "",
	},
	{
		CounterId: 468,
		Instance:  "",
	},
	{
		CounterId: 143,
		Instance:  "vmnic0",
	},
	{
		CounterId: 467,
		Instance:  "vmnic0",
	},
	{
		CounterId: 149,
		Instance:  "vmnic0",
	},
	{
		CounterId: 149,
		Instance:  "",
	},
	{
		CounterId: 470,
		Instance:  "",
	},
	{
		CounterId: 466,
		Instance:  "",
	},
	{
		CounterId: 146,
		Instance:  "",
	},
	{
		CounterId: 465,
		Instance:  "vmnic0",
	},
	{
		CounterId: 461,
		Instance:  "vmnic0",
	},
	{
		CounterId: 466,
		Instance:  "vmnic0",
	},
	{
		CounterId: 146,
		Instance:  "vmnic0",
	},
	{
		CounterId: 464,
		Instance:  "vmnic0",
	},
	{
		CounterId: 148,
		Instance:  "vmnic0",
	},
	{
		CounterId: 460,
		Instance:  "vmnic0",
	},
	{
		CounterId: 468,
		Instance:  "vmnic0",
	},
	{
		CounterId: 147,
		Instance:  "",
	},
	{
		CounterId: 463,
		Instance:  "vmnic0",
	},
	{
		CounterId: 462,
		Instance:  "vmnic0",
	},
	{
		CounterId: 464,
		Instance:  "",
	},
	{
		CounterId: 470,
		Instance:  "vmnic0",
	},
	{
		CounterId: 148,
		Instance:  "",
	},
	{
		CounterId: 462,
		Instance:  "",
	},
	{
		CounterId: 467,
		Instance:  "",
	},
	{
		CounterId: 469,
		Instance:  "",
	},
	{
		CounterId: 147,
		Instance:  "vmnic0",
	},
	{
		CounterId: 159,
		Instance:  "",
	},
	{
		CounterId: 158,
		Instance:  "",
	},
	{
		CounterId: 157,
		Instance:  "",
	},
	{
		CounterId: 503,
		Instance:  "",
	},
	{
		CounterId: 511,
		Instance:  "",
	},
	{
		CounterId: 504,
		Instance:  "",
	},
	{
		CounterId: 501,
		Instance:  "",
	},
	{
		CounterId: 513,
		Instance:  "",
	},
	{
		CounterId: 516,
		Instance:  "",
	},
	{
		CounterId: 507,
		Instance:  "",
	},
	{
		CounterId: 508,
		Instance:  "",
	},
	{
		CounterId: 502,
		Instance:  "",
	},
	{
		CounterId: 348,
		Instance:  "",
	},
	{
		CounterId: 505,
		Instance:  "",
	},
	{
		CounterId: 510,
		Instance:  "",
	},
	{
		CounterId: 512,
		Instance:  "",
	},
	{
		CounterId: 515,
		Instance:  "",
	},
	{
		CounterId: 514,
		Instance:  "",
	},
	{
		CounterId: 506,
		Instance:  "",
	},
	{
		CounterId: 509,
		Instance:  "",
	},
	{
		CounterId: 161,
		Instance:  "vmhba32",
	},
	{
		CounterId: 162,
		Instance:  "vmhba1",
	},
	{
		CounterId: 166,
		Instance:  "vmhba32",
	},
	{
		CounterId: 163,
		Instance:  "vmhba1",
	},
	{
		CounterId: 163,
		Instance:  "vmhba0",
	},
	{
		CounterId: 168,
		Instance:  "",
	},
	{
		CounterId: 167,
		Instance:  "vmhba32",
	},
	{
		CounterId: 162,
		Instance:  "vmhba0",
	},
	{
		CounterId: 164,
		Instance:  "vmhba1",
	},
	{
		CounterId: 167,
		Instance:  "vmhba1",
	},
	{
		CounterId: 167,
		Instance:  "vmhba0",
	},
	{
		CounterId: 162,
		Instance:  "vmhba32",
	},
	{
		CounterId: 164,
		Instance:  "vmhba0",
	},
	{
		CounterId: 166,
		Instance:  "vmhba1",
	},
	{
		CounterId: 166,
		Instance:  "vmhba0",
	},
	{
		CounterId: 165,
		Instance:  "vmhba1",
	},
	{
		CounterId: 165,
		Instance:  "vmhba0",
	},
	{
		CounterId: 164,
		Instance:  "vmhba32",
	},
	{
		CounterId: 161,
		Instance:  "vmhba1",
	},
	{
		CounterId: 161,
		Instance:  "vmhba0",
	},
	{
		CounterId: 163,
		Instance:  "vmhba32",
	},
	{
		CounterId: 165,
		Instance:  "vmhba32",
	},
	{
		CounterId: 520,
		Instance:  "$physDisk",
	},
	{
		CounterId: 523,
		Instance:  "$physDisk",
	},
	{
		CounterId: 193,
		Instance:  "",
	},
	{
		CounterId: 522,
		Instance:  "$physDisk",
	},
	{
		CounterId: 524,
		Instance:  "$physDisk",
	},
	{
		CounterId: 521,
		Instance:  "$physDisk",
	},
	{
		CounterId: 525,
		Instance:  "$physDisk",
	},
	{
		CounterId: 526,
		Instance:  "$physDisk",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 482,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 481,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 488,
		Instance:  "host/vim",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 481,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 480,
		Instance:  "host/vim",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 499,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 496,
		Instance:  "host/system",
	},
	{
		CounterId: 500,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 494,
		Instance:  "host/system",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 491,
		Instance:  "host/system",
	},
	{
		CounterId: 487,
		Instance:  "host/system",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 482,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 499,
		Instance:  "host/system",
	},
	{
		CounterId: 485,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 483,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 484,
		Instance:  "host/system",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 476,
		Instance:  "host/vim",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 493,
		Instance:  "host/system",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 481,
		Instance:  "host/system",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 476,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 499,
		Instance:  "host",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 479,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 484,
		Instance:  "host",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 483,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 481,
		Instance:  "host",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 479,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 500,
		Instance:  "host",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 481,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 484,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 485,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 476,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 482,
		Instance:  "host/vim",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 486,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 492,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 485,
		Instance:  "host/vim",
	},
	{
		CounterId: 483,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 476,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 489,
		Instance:  "host/system",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 499,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 485,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 481,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 497,
		Instance:  "host/system",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 491,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 476,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 485,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 483,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 485,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 499,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 483,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 484,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 480,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 487,
		Instance:  "host/vim",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 484,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 486,
		Instance:  "host/vim",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 483,
		Instance:  "host/system",
	},
	{
		CounterId: 495,
		Instance:  "host/system",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 499,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 488,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 489,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 492,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 499,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 493,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 495,
		Instance:  "host/vim",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 479,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 495,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 482,
		Instance:  "host",
	},
	{
		CounterId: 480,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 480,
		Instance:  "host/user",
	},
	{
		CounterId: 483,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 480,
		Instance:  "host",
	},
	{
		CounterId: 488,
		Instance:  "host/system",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 481,
		Instance:  "host/vim",
	},
	{
		CounterId: 483,
		Instance:  "host/vim",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 490,
		Instance:  "host/vim",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 491,
		Instance:  "host/vim",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 490,
		Instance:  "host/system",
	},
	{
		CounterId: 482,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 479,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel/hostdstats",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 481,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 492,
		Instance:  "host/vim",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 484,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 485,
		Instance:  "host",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 496,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 500,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 485,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 486,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 485,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 483,
		Instance:  "host/user",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 481,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 479,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 481,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 484,
		Instance:  "host/vim",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/var",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 476,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 479,
		Instance:  "host/user",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 487,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 500,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 482,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 155,
		Instance:  "",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/slp",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 480,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 500,
		Instance:  "host/user",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 479,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 480,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 485,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 483,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 476,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 480,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 489,
		Instance:  "host/vim",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 484,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 500,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/wsman",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 497,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 494,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 500,
		Instance:  "host/vim",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 476,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 482,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vsanperfsvc",
	},
	{
		CounterId: 500,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 493,
		Instance:  "host/vim",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 499,
		Instance:  "host/vim",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 479,
		Instance:  "host/vim",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 485,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vobd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 494,
		Instance:  "host/vim",
	},
	{
		CounterId: 496,
		Instance:  "host/vim",
	},
	{
		CounterId: 497,
		Instance:  "host/vim",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmci",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 499,
		Instance:  "host/user",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vimuser/terminal",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/smartd",
	},
	{
		CounterId: 476,
		Instance:  "host",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 476,
		Instance:  "host/system/svmotion",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 483,
		Instance:  "host",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 490,
		Instance:  "host/system/vmotion",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 483,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 484,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 500,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/dhclientrelease",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 479,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 500,
		Instance:  "host/system",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 479,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 480,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 482,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 484,
		Instance:  "host/iofilters",
	},
	{
		CounterId: 476,
		Instance:  "host/system/helper",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 499,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 500,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 482,
		Instance:  "host/system",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/awk",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/dhclient",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 484,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 479,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 483,
		Instance:  "host/system/kernel/tmp",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/head",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/vpxa",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 480,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 481,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 482,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor",
	},
	{
		CounterId: 482,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/usbArbitrator",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/ls",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/pgrep",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 483,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 485,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/probe",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 481,
		Instance:  "host/system/kernel",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/tmp",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vimuser/terminal/ssh",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vmkbacktrace",
	},
	{
		CounterId: 499,
		Instance:  "host/iofilters/iofiltervpd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/hostdCgiServer",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/hbrca",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/init",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 480,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/uwdaemons",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 500,
		Instance:  "host/system/drivers",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/lacpd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 499,
		Instance:  "host/system/kernel/opt",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vmkiscsid",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/upitd",
	},
	{
		CounterId: 481,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 479,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/lbt",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 482,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/aam",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/vsish",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 484,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/likewise",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/snmpd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 482,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe/stats",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/logging",
	},
	{
		CounterId: 484,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 480,
		Instance:  "host/system/kernel/root",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 485,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vimuser/terminal/shell",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 479,
		Instance:  "host/system/ft",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/netcpa",
	},
	{
		CounterId: 499,
		Instance:  "host/iofilters/vmwarevmcrypt",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/sh",
	},
	{
		CounterId: 479,
		Instance:  "host",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 476,
		Instance:  "host/system/kernel/etc",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/vmkdevmgr",
	},
	{
		CounterId: 500,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 476,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/dcui",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/upittraced",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 480,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/nfsgssd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/hostd-probe",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 499,
		Instance:  "host/iofilters/spm",
	},
	{
		CounterId: 500,
		Instance:  "host/system/kernel/iofilters",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/sioc",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/nscd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/net-daemons",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 476,
		Instance:  "host/user",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/boot",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/ntpd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/memScrubber",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/vvold",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/vmkeventd",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/pcscd",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 481,
		Instance:  "host/user",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 480,
		Instance:  "host/system",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/nfcd",
	},
	{
		CounterId: 484,
		Instance:  "host/vim/vmvisor/plugins",
	},
	{
		CounterId: 482,
		Instance:  "host/user",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/sensord",
	},
	{
		CounterId: 500,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 484,
		Instance:  "host/user",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/vvoltraced",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/swapobjd",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/sfcb",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/osfsd",
	},
	{
		CounterId: 485,
		Instance:  "host/user",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 480,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 476,
		Instance:  "host/vim/vmvisor/hostd-probe/stats/logger",
	},
	{
		CounterId: 481,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vimuser",
	},
	{
		CounterId: 482,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 479,
		Instance:  "host/vim/vmvisor/rabbitmqproxy",
	},
	{
		CounterId: 483,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 485,
		Instance:  "host/vim/vmvisor/sfcb_aux",
	},
	{
		CounterId: 499,
		Instance:  "host/vim/vmvisor/vmfstraced",
	},
	{
		CounterId: 541,
		Instance:  "vfc",
	},
}

// ************************************** Cluster Metrics **************************************
var ClusterMetrics = []types.PerfMetricId{
	{
		CounterId: 22,
		Instance:  "",
	},
	{
		CounterId: 2,
		Instance:  "",
	},
	{
		CounterId: 9,
		Instance:  "",
	},
	{
		CounterId: 7,
		Instance:  "",
	},
	{
		CounterId: 8,
		Instance:  "",
	},
	{
		CounterId: 3,
		Instance:  "",
	},
	{
		CounterId: 4,
		Instance:  "",
	},
	{
		CounterId: 15,
		Instance:  "",
	},
	{
		CounterId: 17,
		Instance:  "",
	},
	{
		CounterId: 18,
		Instance:  "",
	},
	{
		CounterId: 19,
		Instance:  "",
	},
	{
		CounterId: 20,
		Instance:  "",
	},
	{
		CounterId: 21,
		Instance:  "",
	},
	{
		CounterId: 6,
		Instance:  "",
	},
	{
		CounterId: 139,
		Instance:  "",
	},
	{
		CounterId: 138,
		Instance:  "",
	},
	{
		CounterId: 107,
		Instance:  "",
	},
	{
		CounterId: 29,
		Instance:  "",
	},
	{
		CounterId: 33,
		Instance:  "",
	},
	{
		CounterId: 37,
		Instance:  "",
	},
	{
		CounterId: 41,
		Instance:  "",
	},
	{
		CounterId: 49,
		Instance:  "",
	},
	{
		CounterId: 90,
		Instance:  "",
	},
	{
		CounterId: 105,
		Instance:  "",
	},
	{
		CounterId: 106,
		Instance:  "",
	},
	{
		CounterId: 27,
		Instance:  "",
	},
	{
		CounterId: 108,
		Instance:  "",
	},
	{
		CounterId: 110,
		Instance:  "",
	},
	{
		CounterId: 111,
		Instance:  "",
	},
	{
		CounterId: 109,
		Instance:  "",
	},
	{
		CounterId: 112,
		Instance:  "",
	},
	{
		CounterId: 25,
		Instance:  "",
	},
	{
		CounterId: 103,
		Instance:  "",
	},
	{
		CounterId: 99,
		Instance:  "",
	},
	{
		CounterId: 30,
		Instance:  "",
	},
	{
		CounterId: 34,
		Instance:  "",
	},
	{
		CounterId: 38,
		Instance:  "",
	},
	{
		CounterId: 42,
		Instance:  "",
	},
	{
		CounterId: 50,
		Instance:  "",
	},
	{
		CounterId: 98,
		Instance:  "",
	},
	{
		CounterId: 26,
		Instance:  "",
	},
	{
		CounterId: 104,
		Instance:  "",
	},
	{
		CounterId: 100,
		Instance:  "",
	},
	{
		CounterId: 31,
		Instance:  "",
	},
	{
		CounterId: 102,
		Instance:  "",
	},
	{
		CounterId: 39,
		Instance:  "",
	},
	{
		CounterId: 43,
		Instance:  "",
	},
	{
		CounterId: 51,
		Instance:  "",
	},
	{
		CounterId: 92,
		Instance:  "",
	},
	{
		CounterId: 24,
		Instance:  "",
	},
	{
		CounterId: 35,
		Instance:  "",
	},
	{
		CounterId: 91,
		Instance:  "",
	},
	{
		CounterId: 153,
		Instance:  "",
	},
	{
		CounterId: 152,
		Instance:  "",
	},
	{
		CounterId: 151,
		Instance:  "",
	},
	{
		CounterId: 150,
		Instance:  "",
	},
	{
		CounterId: 157,
		Instance:  "",
	},
	{
		CounterId: 158,
		Instance:  "",
	},
	{
		CounterId: 159,
		Instance:  "",
	},
	{
		CounterId: 262,
		Instance:  "",
	},
	{
		CounterId: 257,
		Instance:  "",
	},
	{
		CounterId: 258,
		Instance:  "",
	},
	{
		CounterId: 259,
		Instance:  "",
	},
	{
		CounterId: 260,
		Instance:  "",
	},
	{
		CounterId: 261,
		Instance:  "",
	},
	{
		CounterId: 256,
		Instance:  "",
	},
	{
		CounterId: 263,
		Instance:  "",
	},
	{
		CounterId: 264,
		Instance:  "",
	},
	{
		CounterId: 265,
		Instance:  "",
	},
	{
		CounterId: 266,
		Instance:  "",
	},
	{
		CounterId: 267,
		Instance:  "",
	},
	{
		CounterId: 268,
		Instance:  "",
	},
	{
		CounterId: 269,
		Instance:  "",
	},
	{
		CounterId: 270,
		Instance:  "",
	},
	{
		CounterId: 271,
		Instance:  "",
	},
	{
		CounterId: 272,
		Instance:  "",
	},
	{
		CounterId: 273,
		Instance:  "",
	},
	{
		CounterId: 274,
		Instance:  "",
	},
	{
		CounterId: 275,
		Instance:  "",
	},
}

// *************************************** Datastore metrics ****************************************
var DatastoreMetrics = []types.PerfMetricId{
	{
		CounterId: 178,
		Instance:  "",
	},
	{
		CounterId: 188,
		Instance:  "",
	},
	{
		CounterId: 187,
		Instance:  "",
	},
	{
		CounterId: 181,
		Instance:  "",
	},
	{
		CounterId: 180,
		Instance:  "",
	},
	{
		CounterId: 179,
		Instance:  "",
	},
	{
		CounterId: 281,
		Instance:  "",
	},
	{
		CounterId: 281,
		Instance:  "$file",
	},

	{
		CounterId: 282,
		Instance:  "",
	},
	{
		CounterId: 282,
		Instance:  "$file",
	},
	{
		CounterId: 283,
		Instance:  "",
	},
	{
		CounterId: 284,
		Instance:  "",
	},
	{
		CounterId: 284,
		Instance:  "$file",
	},

	{
		CounterId: 288,
		Instance:  "",
	},
	{
		CounterId: 286,
		Instance:  "",
	},
	{
		CounterId: 287,
		Instance:  "",
	},
	{
		CounterId: 287,
		Instance:  "$file",
	},
}

// ********************************************* Resource pool metrics ***********************************
var ResourcePoolMetrics = []types.PerfMetricId{
	{
		CounterId: 6,
		Instance:  "",
	},
	{
		CounterId: 213,
		Instance:  "",
	},
	{
		CounterId: 7,
		Instance:  "",
	},
	{
		CounterId: 8,
		Instance:  "",
	},
	{
		CounterId: 16,
		Instance:  "",
	},
	{
		CounterId: 17,
		Instance:  "",
	},
	{
		CounterId: 18,
		Instance:  "",
	},
	{
		CounterId: 19,
		Instance:  "",
	},
	{
		CounterId: 20,
		Instance:  "",
	},
	{
		CounterId: 22,
		Instance:  "",
	},
	{
		CounterId: 138,
		Instance:  "",
	},
	{
		CounterId: 139,
		Instance:  "",
	},
	{
		CounterId: 112,
		Instance:  "",
	},
	{
		CounterId: 102,
		Instance:  "",
	},
	{
		CounterId: 98,
		Instance:  "",
	},
	{
		CounterId: 29,
		Instance:  "",
	},
	{
		CounterId: 33,
		Instance:  "",
	},
	{
		CounterId: 37,
		Instance:  "",
	},
	{
		CounterId: 41,
		Instance:  "",
	},
	{
		CounterId: 70,
		Instance:  "",
	},
	{
		CounterId: 90,
		Instance:  "",
	},
	{
		CounterId: 108,
		Instance:  "",
	},
	{
		CounterId: 109,
		Instance:  "",
	},
	{
		CounterId: 111,
		Instance:  "",
	},
	{
		CounterId: 214,
		Instance:  "",
	},
	{
		CounterId: 105,
		Instance:  "",
	},
	{
		CounterId: 106,
		Instance:  "",
	},
	{
		CounterId: 107,
		Instance:  "",
	},
	{
		CounterId: 103,
		Instance:  "",
	},
	{
		CounterId: 99,
		Instance:  "",
	},
	{
		CounterId: 30,
		Instance:  "",
	},
	{
		CounterId: 34,
		Instance:  "",
	},
	{
		CounterId: 38,
		Instance:  "",
	},
	{
		CounterId: 42,
		Instance:  "",
	},
	{
		CounterId: 71,
		Instance:  "",
	},
	{
		CounterId: 92,
		Instance:  "",
	},
	{
		CounterId: 104,
		Instance:  "",
	},
	{
		CounterId: 100,
		Instance:  "",
	},
	{
		CounterId: 31,
		Instance:  "",
	},
	{
		CounterId: 35,
		Instance:  "",
	},
	{
		CounterId: 39,
		Instance:  "",
	},
	{
		CounterId: 43,
		Instance:  "",
	},
	{
		CounterId: 72,
		Instance:  "",
	},
	{
		CounterId: 91,
		Instance:  "",
	},
	{
		CounterId: 152,
		Instance:  "",
	},
	{
		CounterId: 153,
		Instance:  "",
	},
	{
		CounterId: 157,
		Instance:  "",
	},
	{
		CounterId: 159,
		Instance:  "",
	},
}

// ********************************************* Datacenter metrics ***********************************
var DatacenterMetrics = []types.PerfMetricId{
	{
		CounterId: 256,
		Instance:  "",
	},
	{
		CounterId: 257,
		Instance:  "",
	},
	{
		CounterId: 258,
		Instance:  "",
	},
	{
		CounterId: 259,
		Instance:  "",
	},
	{
		CounterId: 260,
		Instance:  "",
	},
	{
		CounterId: 261,
		Instance:  "",
	},
	{
		CounterId: 262,
		Instance:  "",
	},
	{
		CounterId: 263,
		Instance:  "",
	},
	{
		CounterId: 264,
		Instance:  "",
	},
	{
		CounterId: 265,
		Instance:  "",
	},
	{
		CounterId: 266,
		Instance:  "",
	},
	{
		CounterId: 267,
		Instance:  "",
	},
	{
		CounterId: 268,
		Instance:  "",
	},
	{
		CounterId: 269,
		Instance:  "",
	},
	{
		CounterId: 270,
		Instance:  "",
	},
	{
		CounterId: 271,
		Instance:  "",
	},
	{
		CounterId: 272,
		Instance:  "",
	},
	{
		CounterId: 273,
		Instance:  "",
	},
	{
		CounterId: 274,
		Instance:  "",
	},
	{
		CounterId: 275,
		Instance:  "",
	},
}
