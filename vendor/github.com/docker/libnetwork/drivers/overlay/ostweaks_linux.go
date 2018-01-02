package overlay

import (
	"io/ioutil"
	"path"
	"strings"

	"github.com/sirupsen/logrus"
)

var sysctlConf = map[string]string{
	"net.ipv4.neigh.default.gc_thresh1": "8192",
	"net.ipv4.neigh.default.gc_thresh2": "49152",
	"net.ipv4.neigh.default.gc_thresh3": "65536",
}

// writeSystemProperty writes the value to a path under /proc/sys as determined from the key.
// For e.g. net.ipv4.ip_forward translated to /proc/sys/net/ipv4/ip_forward.
func writeSystemProperty(key, value string) error {
	keyPath := strings.Replace(key, ".", "/", -1)
	return ioutil.WriteFile(path.Join("/proc/sys", keyPath), []byte(value), 0644)
}

func applyOStweaks() {
	for k, v := range sysctlConf {
		if err := writeSystemProperty(k, v); err != nil {
			logrus.Errorf("error setting the kernel parameter %s = %s, err: %s", k, v, err)
		}
	}
}
