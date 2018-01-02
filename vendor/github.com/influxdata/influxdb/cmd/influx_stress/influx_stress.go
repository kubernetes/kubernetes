package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"

	"github.com/influxdata/influxdb/stress"
	v2 "github.com/influxdata/influxdb/stress/v2"
)

var (
	useV2      = flag.Bool("v2", false, "Use version 2 of stress tool")
	config     = flag.String("config", "", "The stress test file")
	cpuprofile = flag.String("cpuprofile", "", "Write the cpu profile to `filename`")
	db         = flag.String("db", "", "target database within test system for write and query load")
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

	if *useV2 {
		if *config != "" {
			v2.RunStress(*config)
		} else {
			v2.RunStress("stress/v2/iql/file.iql")
		}
	} else {

		c, err := stress.NewConfig(*config)
		if err != nil {
			log.Fatal(err)
			return
		}

		if *db != "" {
			c.Provision.Basic.Database = *db
			c.Write.InfluxClients.Basic.Database = *db
			c.Read.QueryClients.Basic.Database = *db
		}

		w := stress.NewWriter(c.Write.PointGenerators.Basic, &c.Write.InfluxClients.Basic)
		r := stress.NewQuerier(&c.Read.QueryGenerators.Basic, &c.Read.QueryClients.Basic)
		s := stress.NewStressTest(&c.Provision.Basic, w, r)

		bw := stress.NewBroadcastChannel()
		bw.Register(c.Write.InfluxClients.Basic.BasicWriteHandler)
		bw.Register(o.HTTPHandler("write"))

		br := stress.NewBroadcastChannel()
		br.Register(c.Read.QueryClients.Basic.BasicReadHandler)
		br.Register(o.HTTPHandler("read"))

		s.Start(bw.Handle, br.Handle)

	}
}
