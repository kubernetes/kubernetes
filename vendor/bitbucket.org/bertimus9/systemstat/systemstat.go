// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

package systemstat

import (
	"time"
)

// CPUSample is an object that represents the breakdown of time spent by the
// CPU in various types of tasks. Two CPUSamples are required to find the
// average usage over time, represented by the CPUAverage object. The CPUSample
// is taken from the line "cpu" from /proc/stat in the Linux kernel.
//
// Summarized from the proc(5) man page:
// /proc/stat :
//        kernel/system  statistics.   Varies  with  architecture.
type CPUSample struct {
	User    uint64    // time spent in user mode
	Nice    uint64    // time spent in user mode with low priority (nice)
	System  uint64    // time spent in system mode
	Idle    uint64    // time spent in the idle task
	Iowait  uint64    // time spent waiting for I/O to complete (since Linux 2.5.41)
	Irq     uint64    // time spent servicing  interrupts  (since  2.6.0-test4)
	SoftIrq uint64    // time spent servicing softirqs (since 2.6.0-test4)
	Steal   uint64    // time spent in other OSes when running in a virtualized environment
	Guest   uint64    // time spent running a virtual CPU for guest operating systems under the control of the Linux kernel.
	Name    string    // name of the line in /proc/stat; cpu, cpu1, etc
	Time    time.Time // when the sample was taken
	Total   uint64    // total of all time fields
}

type ProcCPUSample struct {
	User         float64   // time spent in user mode
	System       float64   // time spent in system mode
	Time         time.Time // when the sample was taken
	Total        float64   // total of all time fields
	ProcMemUsedK int64
}

type ProcCPUAverage struct {
	UserPct            float64   // time spent in user mode
	SystemPct          float64   // time spent in system mode
	TotalPct           float64   // total of all time fields
	PossiblePct        float64   // total of all time fields
	CumulativeTotalPct float64   // total of all time throughout process life
	Time               time.Time // when the sample was taken
	Seconds            float64   // how many seconds between the two samples
}

// SimpleCPUAverage is an object that represents the average cpu usage over a
// time period. It is calculated by taking the difference between two
// CPUSamples (whose units are clock ticks), dividing by the number of elapsed
// ticks between the samples, and converting to a percent. It is a simplified version of the CPUAverage in that it only accounts for time in the Idle task and all other time (Busy).
type SimpleCPUAverage struct {
	BusyPct float64 // percent of time spent by CPU performing all non-idle tasks
	IdlePct float64 // percent of time spent by CPU in the idle task
}

// CPUAverage is an object that represents the average cpu usage over a
// time period. It is calculated by taking the difference between two
// CPUSamples (whose units are clock ticks), dividing by the number of elapsed
// ticks between the samples, and converting to a percent.
type CPUAverage struct {
	UserPct    float64
	NicePct    float64
	SystemPct  float64
	IdlePct    float64
	IowaitPct  float64
	IrqPct     float64
	SoftIrqPct float64
	StealPct   float64
	GuestPct   float64
	Time       time.Time
	Seconds    float64 // how many seconds between the two samples
}

type MemSample struct {
	Buffers   uint64
	Cached    uint64
	MemTotal  uint64
	MemUsed   uint64
	MemFree   uint64
	SwapTotal uint64
	SwapUsed  uint64
	SwapFree  uint64
	Time      time.Time
}

type LoadAvgSample struct {
	One     float64
	Five    float64
	Fifteen float64
	Time    time.Time
}

type UptimeSample struct {
	Uptime float64
	Time   time.Time
}

// GetCPUAverage returns the average cpu usage between two CPUSamples.
func GetCPUAverage(first CPUSample, second CPUSample) CPUAverage {
	return getCPUAverage(first, second)
}

// GetSimpleCPUAverage returns an aggregated average cpu usage between two CPUSamples.
func GetSimpleCPUAverage(first CPUSample, second CPUSample) SimpleCPUAverage {
	return getSimpleCPUAverage(first, second)
}

// GetProcCPUAverage returns the average cpu usage of this running process
func GetProcCPUAverage(first ProcCPUSample, second ProcCPUSample, procUptime float64) (avg ProcCPUAverage) {
	return getProcCPUAverage(first, second, procUptime)
}

// GetCPUSample takes a snapshot of kernel statistics from the /proc/stat file.
func GetCPUSample() (samp CPUSample) {
	return getCPUSample("/proc/stat")
}

// GetProcCPUSample takes a snapshot of kernel statistics from the /proc/stat file.
func GetProcCPUSample() (samp ProcCPUSample) {
	return getProcCPUSample()
}

// GetUptime takes a snapshot of load info from the /proc/loadavg file.
func GetUptime() (samp UptimeSample) {
	return getUptime("/proc/uptime")
}

// GetLoadAvgSample takes a snapshot of load info from the /proc/loadavg file.
func GetLoadAvgSample() (samp LoadAvgSample) {
	return getLoadAvgSample("/proc/loadavg")
}

// GetMemSample takes a snapshot of memory info from the /proc/meminfo file.
func GetMemSample() (samp MemSample) {
	return getMemSample("/proc/meminfo")
}
