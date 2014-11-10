package core

import (
	"flag"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/scaler/actuator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scaler/aggregator"
	"github.com/golang/glog"
)

type realAutoScaler struct {
}

var argHousekeepingTick = flag.Duration("housekeeping", 10*time.Second, "Housekeeping duration.")

func (self *realAutoScaler) AutoScale() error {
	agg := aggregator.New()
	act := actuator.New()

	for {
		_, err := agg.GetClusterInfo()
		if err != nil {
			glog.Errorf("Failed to get cluster node information from aggregator - %q", err)
			continue
		}
		newNodes := []actuator.NodeShape{}
		// apply policy
		// Create new nodes if needed
		if len(newNodes) > 0 {
			err := act.CreateNewNodes(newNodes)
			if err != nil {
				glog.Errorf("Failed to create a new node - %q", err)
			}
		}
		// Sleep for housekeeping duration.
		time.Sleep(*argHousekeepingTick)
	}
	return nil
}

func New() Scaler {
	return &realAutoScaler{}
}
