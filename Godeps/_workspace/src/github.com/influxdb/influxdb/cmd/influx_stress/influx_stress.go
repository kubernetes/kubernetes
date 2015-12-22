package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/influxdb/influxdb/stress"
)

var (
	config     = flag.String("config", "", "The stress test file")
	cpuprofile = flag.String("cpuprofile", "", "Write the cpu profile to `filename`")
)

func main() {
	o := stress.NewOutputConfig()
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Println(err)
			return
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	c, err := stress.NewConfig(*config)
	if err != nil {
		fmt.Println(err)
		return
	}

	w := stress.NewWriter(&c.Write.PointGenerators.Basic, &c.Write.InfluxClients.Basic)
	r := stress.NewQuerier(&c.Read.QueryGenerators.Basic, &c.Read.QueryClients.Basic)
	s := stress.NewStressTest(&c.Provision.Basic, w, r)

	bw := stress.NewBroadcastChannel()
	bw.Register(c.Write.InfluxClients.Basic.BasicWriteHandler)
	bw.Register(o.HTTPHandler("write"))

	br := stress.NewBroadcastChannel()
	br.Register(c.Read.QueryClients.Basic.BasicReadHandler)
	br.Register(o.HTTPHandler("read"))

	s.Start(bw.Handle, br.Handle)

	return

}
