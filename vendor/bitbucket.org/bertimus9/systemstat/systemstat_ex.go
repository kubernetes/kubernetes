// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

//go:build !linux

package systemstat

import (
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
	notImplemented("getProcCPUSample")
	s.Time = time.Now()
	return
}

func getCPUSample(procfile string) (samp CPUSample) {
	notImplemented("getCPUSample")
	samp.Time = time.Now()
	return
}
