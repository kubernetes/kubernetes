package procfs

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
)

// ProcLimits represents the soft limits for each of the process's resource
// limits. For more information see getrlimit(2):
// http://man7.org/linux/man-pages/man2/getrlimit.2.html.
type ProcLimits struct {
	// CPU time limit in seconds.
	CPUTime int
	// Maximum size of files that the process may create.
	FileSize int
	// Maximum size of the process's data segment (initialized data,
	// uninitialized data, and heap).
	DataSize int
	// Maximum size of the process stack in bytes.
	StackSize int
	// Maximum size of a core file.
	CoreFileSize int
	// Limit of the process's resident set in pages.
	ResidentSet int
	// Maximum number of processes that can be created for the real user ID of
	// the calling process.
	Processes int
	// Value one greater than the maximum file descriptor number that can be
	// opened by this process.
	OpenFiles int
	// Maximum number of bytes of memory that may be locked into RAM.
	LockedMemory int
	// Maximum size of the process's virtual memory address space in bytes.
	AddressSpace int
	// Limit on the combined number of flock(2) locks and fcntl(2) leases that
	// this process may establish.
	FileLocks int
	// Limit of signals that may be queued for the real user ID of the calling
	// process.
	PendingSignals int
	// Limit on the number of bytes that can be allocated for POSIX message
	// queues for the real user ID of the calling process.
	MsqqueueSize int
	// Limit of the nice priority set using setpriority(2) or nice(2).
	NicePriority int
	// Limit of the real-time priority set using sched_setscheduler(2) or
	// sched_setparam(2).
	RealtimePriority int
	// Limit (in microseconds) on the amount of CPU time that a process
	// scheduled under a real-time scheduling policy may consume without making
	// a blocking system call.
	RealtimeTimeout int
}

const (
	limitsFields    = 3
	limitsUnlimited = "unlimited"
)

var (
	limitsDelimiter = regexp.MustCompile("  +")
)

// NewLimits returns the current soft limits of the process.
func (p Proc) NewLimits() (ProcLimits, error) {
	f, err := os.Open(p.path("limits"))
	if err != nil {
		return ProcLimits{}, err
	}
	defer f.Close()

	var (
		l = ProcLimits{}
		s = bufio.NewScanner(f)
	)
	for s.Scan() {
		fields := limitsDelimiter.Split(s.Text(), limitsFields)
		if len(fields) != limitsFields {
			return ProcLimits{}, fmt.Errorf(
				"couldn't parse %s line %s", f.Name(), s.Text())
		}

		switch fields[0] {
		case "Max cpu time":
			l.CPUTime, err = parseInt(fields[1])
		case "Max file size":
			l.FileSize, err = parseInt(fields[1])
		case "Max data size":
			l.DataSize, err = parseInt(fields[1])
		case "Max stack size":
			l.StackSize, err = parseInt(fields[1])
		case "Max core file size":
			l.CoreFileSize, err = parseInt(fields[1])
		case "Max resident set":
			l.ResidentSet, err = parseInt(fields[1])
		case "Max processes":
			l.Processes, err = parseInt(fields[1])
		case "Max open files":
			l.OpenFiles, err = parseInt(fields[1])
		case "Max locked memory":
			l.LockedMemory, err = parseInt(fields[1])
		case "Max address space":
			l.AddressSpace, err = parseInt(fields[1])
		case "Max file locks":
			l.FileLocks, err = parseInt(fields[1])
		case "Max pending signals":
			l.PendingSignals, err = parseInt(fields[1])
		case "Max msgqueue size":
			l.MsqqueueSize, err = parseInt(fields[1])
		case "Max nice priority":
			l.NicePriority, err = parseInt(fields[1])
		case "Max realtime priority":
			l.RealtimePriority, err = parseInt(fields[1])
		case "Max realtime timeout":
			l.RealtimeTimeout, err = parseInt(fields[1])
		}
		if err != nil {
			return ProcLimits{}, err
		}
	}

	return l, s.Err()
}

func parseInt(s string) (int, error) {
	if s == limitsUnlimited {
		return -1, nil
	}
	i, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("couldn't parse value %s: %s", s, err)
	}
	return int(i), nil
}
