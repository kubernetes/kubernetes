package api

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

// TODO(phase2) use componentconfig
// we need some params for testing etc, let's keep these hidden for now
func GetEnvParams() map[string]string {

	envParams := map[string]string{
		// TODO(phase1+): Mode prefix and host_pki_path to another place as constants, and use them everywhere
		// Right now they're used here and there, but not consequently
		"kubernetes_dir":     "/etc/kubernetes",
		"host_pki_path":      "/etc/kubernetes/pki",
		"host_etcd_path":     "/var/lib/etcd",
		"hyperkube_image":    "",
		"discovery_image":    fmt.Sprintf("gcr.io/google_containers/kube-discovery-%s:%s", runtime.GOARCH, "1.0"),
		"etcd_image":         "",
		"component_loglevel": "--v=4",
	}

	for k := range envParams {
		if v := os.Getenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k))); v != "" {
			envParams[k] = v
		}
	}

	return envParams
}
