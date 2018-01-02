package main

import "fmt"

const (
	RLIMIT_CPU        = iota // CPU time in sec
	RLIMIT_FSIZE             // Maximum filesize
	RLIMIT_DATA              // max data size
	RLIMIT_STACK             // max stack size
	RLIMIT_CORE              // max core file size
	RLIMIT_RSS               // max resident set size
	RLIMIT_NPROC             // max number of processes
	RLIMIT_NOFILE            // max number of open files
	RLIMIT_MEMLOCK           // max locked-in-memory address space
	RLIMIT_AS                // address space limit
	RLIMIT_LOCKS             // maximum file locks held
	RLIMIT_SIGPENDING        // max number of pending signals
	RLIMIT_MSGQUEUE          // maximum bytes in POSIX mqueues
	RLIMIT_NICE              // max nice prio allowed to raise to
	RLIMIT_RTPRIO            // maximum realtime priority
	RLIMIT_RTTIME            // timeout for RT tasks in us
)

var rlimitMap = map[string]int{
	"RLIMIT_CPU":        RLIMIT_CPU,
	"RLIMIT_FSIZE":      RLIMIT_FSIZE,
	"RLIMIT_DATA":       RLIMIT_DATA,
	"RLIMIT_STACK":      RLIMIT_STACK,
	"RLIMIT_CORE":       RLIMIT_CORE,
	"RLIMIT_RSS":        RLIMIT_RSS,
	"RLIMIT_NPROC":      RLIMIT_NPROC,
	"RLIMIT_NOFILE":     RLIMIT_NOFILE,
	"RLIMIT_MEMLOCK":    RLIMIT_MEMLOCK,
	"RLIMIT_AS":         RLIMIT_AS,
	"RLIMIT_LOCKS":      RLIMIT_LOCKS,
	"RLIMIT_SIGPENDING": RLIMIT_SIGPENDING,
	"RLIMIT_MSGQUEUE":   RLIMIT_MSGQUEUE,
	"RLIMIT_NICE":       RLIMIT_NICE,
	"RLIMIT_RTPRIO":     RLIMIT_RTPRIO,
	"RLIMIT_RTTIME":     RLIMIT_RTTIME,
}

func strToRlimit(key string) (int, error) {
	rl, ok := rlimitMap[key]
	if !ok {
		return 0, fmt.Errorf("wrong rlimit value: %s", key)
	}
	return rl, nil
}
