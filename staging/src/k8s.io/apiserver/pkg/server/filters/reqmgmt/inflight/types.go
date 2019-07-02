package inflight

import (
	"time"

	"k8s.io/apiserver/pkg/server/filters/fq"
)

// Queue is an array of packets with additional metadata required for
// the FQScheduler
type Queue struct {
	fq.Queue
	// TODO(aaron-prindle) verify other info we might need...
	// Priority    PriorityBand
	// SharedQuota int
	// Index       int
}

// required for the functionality FQFilter
type Packet struct {
	fq.Packet
	DequeueChannel chan bool
	Seq            int
	EnqueueTime    time.Time
}
