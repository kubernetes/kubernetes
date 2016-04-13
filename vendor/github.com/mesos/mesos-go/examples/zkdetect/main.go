package main

import (
	"flag"
	"fmt"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	_ "github.com/mesos/mesos-go/detector/zoo"
	mesos "github.com/mesos/mesos-go/mesosproto"
)

type zkListener struct{}

func (l *zkListener) OnMasterChanged(info *mesos.MasterInfo) {
	if info == nil {
		log.Infoln("master lost")
	} else {
		log.Infof("master changed: %s", masterString(info))
	}
}

func (l *zkListener) UpdatedMasters(all []*mesos.MasterInfo) {
	for i, info := range all {
		log.Infof("master (%d): %s", i, masterString(info))
	}
}

func masterString(info *mesos.MasterInfo) string {
	return fmt.Sprintf("Id %v Ip %v Hostname %v Port %v Version %v Pid %v",
		info.GetId(), info.GetIp(), info.GetHostname(), info.GetPort(), info.GetVersion(), info.GetPid())
}

func main() {
	masters := flag.String("masters", "zk://localhost:2181/mesos", "ZK Mesos masters URI")
	flag.Parse()

	log.Infof("creating ZK detector for %q", *masters)

	m, err := detector.New(*masters)
	if err != nil {
		log.Fatalf("failed to create ZK listener for Mesos masters: %v", err)
	}

	log.Info("created ZK detector")
	err = m.Detect(&zkListener{})
	if err != nil {
		log.Fatalf("failed to register ZK listener: %v", err)
	}

	log.Info("registered ZK listener")
	select {} // never stop
}
