package etcd

import (
	"math/rand"
	"strings"
)

type Cluster struct {
	Leader   string   `json:"leader"`
	Machines []string `json:"machines"`
	picked   int
}

func NewCluster(machines []string) *Cluster {
	// if an empty slice was sent in then just assume HTTP 4001 on localhost
	if len(machines) == 0 {
		machines = []string{"http://127.0.0.1:4001"}
	}

	// default leader and machines
	return &Cluster{
		Leader:   "",
		Machines: machines,
		picked:   rand.Intn(len(machines)),
	}
}

func (cl *Cluster) failure()     { cl.picked = rand.Intn(len(cl.Machines)) }
func (cl *Cluster) pick() string { return cl.Machines[cl.picked] }

func (cl *Cluster) updateFromStr(machines string) {
	cl.Machines = strings.Split(machines, ",")
	for i := range cl.Machines {
		cl.Machines[i] = strings.TrimSpace(cl.Machines[i])
	}
	cl.picked = rand.Intn(len(cl.Machines))
}
