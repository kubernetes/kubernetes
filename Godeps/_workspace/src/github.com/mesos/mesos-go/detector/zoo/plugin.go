package zoo

import (
	"github.com/mesos/mesos-go/detector"
)

func init() {
	detector.Register("zk://", detector.PluginFactory(func(spec string) (detector.Master, error) {
		return NewMasterDetector(spec)
	}))
}
