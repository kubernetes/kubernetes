package mesosutil

import (
	"os"
	"os/exec"
	"strings"

	log "github.com/golang/glog"
)

//TODO(jdef) copied from kubernetes/pkg/util/node.go
func GetHostname(hostnameOverride string) string {
	hostname := hostnameOverride
	if hostname == "" {
		// Note: We use exec here instead of os.Hostname() because we
		// want the FQDN, and this is the easiest way to get it.
		fqdn, err := exec.Command("hostname", "-f").Output()
		if err != nil || len(fqdn) == 0 {
			log.Errorf("Couldn't determine hostname fqdn, failing back to hostname: %v", err)
			hostname, err = os.Hostname()
			if err != nil {
				log.Fatalf("Error getting hostname: %v", err)
			}
		} else {
			hostname = string(fqdn)
		}
	}
	return strings.TrimSpace(hostname)
}
