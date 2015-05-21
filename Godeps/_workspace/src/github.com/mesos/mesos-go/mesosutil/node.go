package mesosutil

import (
	"os/exec"
	"strings"

	log "github.com/golang/glog"
)

//TODO(jdef) copied from kubernetes/pkg/util/node.go
func GetHostname(hostnameOverride string) string {
	hostname := []byte(hostnameOverride)
	if string(hostname) == "" {
		// Note: We use exec here instead of os.Hostname() because we
		// want the FQDN, and this is the easiest way to get it.
		fqdn, err := exec.Command("hostname", "-f").Output()
		if err != nil {
			log.Fatalf("Couldn't determine hostname: %v", err)
		}
		hostname = fqdn
	}
	return strings.TrimSpace(string(hostname))
}
