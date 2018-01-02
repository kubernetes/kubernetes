// +build !windows

package containerd

import (
	"runtime"
)

const (
	defaultRoot    = "/var/lib/containerd-test"
	defaultState   = "/run/containerd-test"
	defaultAddress = "/run/containerd-test/containerd.sock"
)

var (
	testImage string
)

func platformTestSetup(client *Client) error {
	return nil
}

func init() {
	switch runtime.GOARCH {
	case "386":
		testImage = "docker.io/i386/alpine:latest"
	case "arm":
		testImage = "docker.io/arm32v6/alpine:latest"
	case "arm64":
		testImage = "docker.io/arm64v8/alpine:latest"
	case "ppc64le":
		testImage = "docker.io/ppc64le/alpine:latest"
	case "s390x":
		testImage = "docker.io/s390x/alpine:latest"
	default:
		testImage = "docker.io/library/alpine:latest"
	}
}
