package etcd

import (
	"math/rand"
	"strings"
	"sync"
)

type Cluster struct {
	Leader   string   `json:"leader"`
	Machines []string `json:"machines"`
	picked   int
	mu       sync.RWMutex
}

func NewCluster(machines []string) *Cluster {
	// if an empty slice was sent in then just assume HTTP 4001 on localhost
	if len(machines) == 0 {
		machines = []string{"http://127.0.0.1:4001"}
	}

	machines = shuffleStringSlice(machines)
	logger.Debug("Shuffle cluster machines", machines)
	// default leader and machines
	return &Cluster{
		Leader:   "",
		Machines: machines,
		picked:   rand.Intn(len(machines)),
	}
}

func (cl *Cluster) failure() {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	cl.picked = (cl.picked + 1) % len(cl.Machines)
}

func (cl *Cluster) pick() string {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	return cl.Machines[cl.picked]
}

func (cl *Cluster) updateFromStr(machines string) {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	cl.Machines = strings.Split(machines, ",")
	for i := range cl.Machines {
		cl.Machines[i] = strings.TrimSpace(cl.Machines[i])
	}
	cl.Machines = shuffleStringSlice(cl.Machines)
	cl.picked = rand.Intn(len(cl.Machines))
}
