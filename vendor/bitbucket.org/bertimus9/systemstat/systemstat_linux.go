// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

// +build linux

package systemstat

import (
	"bufio"
	"bytes"
	"io"
	"io/ioutil"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func getUptime(procfile string) (uptime UptimeSample) {
	// read in whole uptime file with cpu usage information ;"/proc/uptime"
	contents, err := ioutil.ReadFile(procfile)
	uptime.Time = time.Now()
	if err != nil {
		return
	}

	reader := bufio.NewReader(bytes.NewBuffer(contents))
	line, _, err := reader.ReadLine()
	fields := strings.Fields(string(line))

	val, numerr := strconv.ParseFloat(fields[0], 64)
	if numerr != nil {
		return
	}
	uptime.Uptime = val

	return
}

func getLoadAvgSample(procfile string) (samp LoadAvgSample) {
	// read in whole loadavg file with cpu usage information ;"/proc/loadavg"
	contents, err := ioutil.ReadFile(procfile)
	samp.Time = time.Now()
	if err != nil {
		return
	}

	reader := bufio.NewReader(bytes.NewBuffer(contents))
	line, _, err := reader.ReadLine()
	fields := strings.Fields(string(line))
	for i := 0; i < 3; i++ {
		val, numerr := strconv.ParseFloat(fields[i], 64)
		if numerr != nil {
			return
		}
		switch i {
		case 0:
			samp.One = val
		case 1:
			samp.Five = val
		case 2:
			samp.Fifteen = val
		}
	}

	return
}

func getMemSample(procfile string) (samp MemSample) {
	want := map[string]bool{
		"Buffers:":   true,
		"Cached:":    true,
		"MemTotal:":  true,
		"MemFree:":   true,
		"MemUsed:":   true,
		"SwapTotal:": true,
		"SwapFree:":  true,
		"SwapUsed:":  true}

	// read in whole meminfo file with cpu usage information ;"/proc/meminfo"
	contents, err := ioutil.ReadFile(procfile)
	samp.Time = time.Now()
	if err != nil {
		return
	}

	reader := bufio.NewReader(bytes.NewBuffer(contents))
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		}

		fields := strings.Fields(string(line))
		fieldName := fields[0]

		_, ok := want[fieldName]
		if ok && len(fields) == 3 {
			val, numerr := strconv.ParseUint(fields[1], 10, 64)
			if numerr != nil {
				return
			}
			switch fieldName {
			case "Buffers:":
				samp.Buffers = val
			case "Cached:":
				samp.Cached = val
			case "MemTotal:":
				samp.MemTotal = val
			case "MemFree:":
				samp.MemFree = val
			case "SwapTotal:":
				samp.SwapTotal = val
			case "SwapFree:":
				samp.SwapFree = val
			}
		}
	}
	samp.MemUsed = samp.MemTotal - samp.MemFree
	samp.SwapUsed = samp.SwapTotal - samp.SwapFree
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
	// read in whole proc file with cpu usage information ; "/proc/stat"
	contents, err := ioutil.ReadFile(procfile)
	samp.Time = time.Now()
	if err != nil {
		return
	}

	reader := bufio.NewReader(bytes.NewBuffer(contents))
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		}

		fields := strings.Fields(string(line))

		if len(fields) > 0 {
			fieldName := fields[0]
			if fieldName == "cpu" {
				parseCPUFields(fields, &samp)
			}
		}
	}
	return
}
