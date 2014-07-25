// The job controller is responsible for watching jobs, running them,
// and monitoring them.
package main

import (
	"flag"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/job"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	etcdServerList util.StringList
	master         = flag.String("master", "", "The address of the Kubernetes API server")
)

func init() {
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	if len(etcdServerList) == 0 || len(*master) == 0 {
		glog.Fatal("usage: controller-manager -master <master>")
	}

	jobController := job.MakeJobController(client.New("http://"+*master, nil))

	jobController.Run(10 * time.Second)
	select {}
}
