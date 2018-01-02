package libnetwork

import "github.com/docker/docker/pkg/reexec"

type setKeyData struct {
	ContainerID string
	Key         string
}

func init() {
	reexec.Register("libnetwork-setkey", processSetKeyReexec)
}
