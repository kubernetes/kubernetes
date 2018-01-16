// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

// +build !linux

package systemstat

import (
	"syscall"
	"time"
)

func getUptime(procfile string) (uptime UptimeSample) {
	notImplemented("getUptime")
	uptime.Time = time.Now()
	return
}

func getLoadAvgSample(procfile string) (samp LoadAvgSample) {
	notImplemented("getLoadAvgSample")
	samp.Time = time.Now()
	return
}

func getMemSample(procfile string) (samp MemSample) {
	notImplemented("getMemSample")
	samp.Time = time.Now()
	return
}

func getProcCPUSample() (s ProcCPUSample) {
	var processInfo syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &processInfo)

	s.Time = time.Now()
	s.ProcMemUsedK = int64(processInfo.Maxrss)
	s.User = float64(processInfo.Utime.Usec)/1000000 + float64(processInfo.Utime.Sec)
	s.System = float64(processInfo.Stime.Usec)/1000000 + float64(processInfo.Stime.Sec)
	s.Total = s.User + s.System

	return
}

func getCPUSample(procfile string) (samp CPUSample) {
	notImplemented("getCPUSample")
	samp.Time = time.Now()
	return
}
