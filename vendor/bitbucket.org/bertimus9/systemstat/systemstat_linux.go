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
	"log"
	"runtime"
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

func getSimpleCPUAverage(first CPUSample, second CPUSample) (avg SimpleCPUAverage) {
	//walltimediff := second.Time.Sub(first.Time)
	//dT := float64(first.Total - second.Total)

	dI := float64(second.Idle - first.Idle)
	dTot := float64(second.Total - first.Total)
	avg.IdlePct = dI / dTot * 100
	avg.BusyPct = (dTot - dI) * 100 / dTot
	//log.Printf("cpu idle ticks %f, total ticks %f, idle pct %f, busy pct %f\n", dI, dTot, avg.IdlePct, avg.BusyPct)
	return
}

func subtractAndConvertTicks(first uint64, second uint64) float64 {
	return float64(first - second)
}

func getCPUAverage(first CPUSample, second CPUSample) (avg CPUAverage) {
	dTot := float64(second.Total - first.Total)
	invQuotient := 100.00 / dTot

	avg.UserPct = subtractAndConvertTicks(second.User, first.User) * invQuotient
	avg.NicePct = subtractAndConvertTicks(second.Nice, first.Nice) * invQuotient
	avg.SystemPct = subtractAndConvertTicks(second.System, first.System) * invQuotient
	avg.IdlePct = subtractAndConvertTicks(second.Idle, first.Idle) * invQuotient
	avg.IowaitPct = subtractAndConvertTicks(second.Iowait, first.Iowait) * invQuotient
	avg.IrqPct = subtractAndConvertTicks(second.Irq, first.Irq) * invQuotient
	avg.SoftIrqPct = subtractAndConvertTicks(second.SoftIrq, first.SoftIrq) * invQuotient
	avg.StealPct = subtractAndConvertTicks(second.Steal, first.Steal) * invQuotient
	avg.GuestPct = subtractAndConvertTicks(second.Guest, first.Guest) * invQuotient
	avg.Time = second.Time
	avg.Seconds = second.Time.Sub(first.Time).Seconds()
	return
}

func getProcCPUAverage(first ProcCPUSample, second ProcCPUSample, procUptime float64) (avg ProcCPUAverage) {
	dT := second.Time.Sub(first.Time).Seconds()

	avg.UserPct = 100 * (second.User - first.User) / dT
	avg.SystemPct = 100 * (second.System - first.System) / dT
	avg.TotalPct = 100 * (second.Total - first.Total) / dT
	avg.PossiblePct = 100.0 * float64(runtime.NumCPU())
	avg.CumulativeTotalPct = 100 * second.Total / procUptime
	avg.Time = second.Time
	avg.Seconds = dT
	return
}

func parseCPUFields(fields []string, stat *CPUSample) {
	numFields := len(fields)
	stat.Name = fields[0]
	for i := 1; i < numFields; i++ {
		val, numerr := strconv.ParseUint(fields[i], 10, 64)
		if numerr != nil {
			log.Println("systemstat.parseCPUFields(): Error parsing (field, value): ", i, fields[i])
		}
		stat.Total += val
		switch i {
		case 1:
			stat.User = val
		case 2:
			stat.Nice = val
		case 3:
			stat.System = val
		case 4:
			stat.Idle = val
		case 5:
			stat.Iowait = val
		case 6:
			stat.Irq = val
		case 7:
			stat.SoftIrq = val
		case 8:
			stat.Steal = val
		case 9:
			stat.Guest = val
		}
	}
}
