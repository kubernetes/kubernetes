package systemstat

import (
	"log"
	"runtime"
	"strconv"
)

func notImplemented(fn string) {
	log.Printf("systemstat/%s is not implemented for this OS: %s\n", fn, runtime.GOOS)
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
