package swarm // import "github.com/docker/docker/api/types/swarm"

import (
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
)

// DNSConfig specifies DNS related configurations in resolver configuration file (resolv.conf)
// Detailed documentation is available in:
// http://man7.org/linux/man-pages/man5/resolv.conf.5.html
// `nameserver`, `search`, `options` have been supported.
// TODO: `domain` is not supported yet.
type DNSConfig struct {
	// Nameservers specifies the IP addresses of the name servers
	Nameservers []string `json:",omitempty"`
	// Search specifies the search list for host-name lookup
	Search []string `json:",omitempty"`
	// Options allows certain internal resolver variables to be modified
	Options []string `json:",omitempty"`
}

// SELinuxContext contains the SELinux labels of the container.
type SELinuxContext struct {
	Disable bool

	User  string
	Role  string
	Type  string
	Level string
}

// CredentialSpec for managed service account (Windows only)
type CredentialSpec struct {
	Config   string
	File     string
	Registry string
}

// Privileges defines the security options for the container.
type Privileges struct {
	CredentialSpec *CredentialSpec
	SELinuxContext *SELinuxContext
}

// ContainerSpec represents the spec of a container.
type ContainerSpec struct {
	Image           string                  `json:",omitempty"`
	Labels          map[string]string       `json:",omitempty"`
	Command         []string                `json:",omitempty"`
	Args            []string                `json:",omitempty"`
	Hostname        string                  `json:",omitempty"`
	Env             []string                `json:",omitempty"`
	Dir             string                  `json:",omitempty"`
	User            string                  `json:",omitempty"`
	Groups          []string                `json:",omitempty"`
	Privileges      *Privileges             `json:",omitempty"`
	Init            *bool                   `json:",omitempty"`
	StopSignal      string                  `json:",omitempty"`
	TTY             bool                    `json:",omitempty"`
	OpenStdin       bool                    `json:",omitempty"`
	ReadOnly        bool                    `json:",omitempty"`
	Mounts          []mount.Mount           `json:",omitempty"`
	StopGracePeriod *time.Duration          `json:",omitempty"`
	Healthcheck     *container.HealthConfig `json:",omitempty"`
	// The format of extra hosts on swarmkit is specified in:
	// http://man7.org/linux/man-pages/man5/hosts.5.html
	//    IP_address canonical_hostname [aliases...]
	Hosts     []string            `json:",omitempty"`
	DNSConfig *DNSConfig          `json:",omitempty"`
	Secrets   []*SecretReference  `json:",omitempty"`
	Configs   []*ConfigReference  `json:",omitempty"`
	Isolation container.Isolation `json:",omitempty"`
	Sysctls   map[string]string   `json:",omitempty"`
}
