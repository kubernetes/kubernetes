// The build controller is responsible for running jobs for builds
// watching them, and updating build state.
package main

import (
	"flag"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/build"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	master = flag.String("master", "", "The address of the Kubernetes API server")
)

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	if len(*master) == 0 {
		glog.Fatal("usage: controller-manager -etcd_servers <servers> -master <master>")
	}

	buildController := build.MakeBuildController(client.New("http://"+*master, nil))

	buildController.Run(10 * time.Second)
	select {}
}
