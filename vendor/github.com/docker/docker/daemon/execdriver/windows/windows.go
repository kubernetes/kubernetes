// +build windows

package windows

import (
	"fmt"
	"strings"
	"sync"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/pkg/parsers"
)

// This is a daemon development variable only and should not be
// used for running production containers on Windows.
var dummyMode bool

// This allows the daemon to terminate containers rather than shutdown
var terminateMode bool

var (
	DriverName = "Windows 1854"
	Version    = dockerversion.VERSION + " " + dockerversion.GITCOMMIT
)

type activeContainer struct {
	command *execdriver.Command
}

type driver struct {
	root             string
	initPath         string
	activeContainers map[string]*activeContainer
	sync.Mutex
}

func (d *driver) Name() string {
	return fmt.Sprintf("%s %s", DriverName, Version)
}

func NewDriver(root, initPath string, options []string) (*driver, error) {

	for _, option := range options {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil {
			return nil, err
		}
		key = strings.ToLower(key)
		switch key {

		case "dummy":
			switch val {
			case "1":
				dummyMode = true
				logrus.Warn("Using dummy mode in Windows exec driver. This is for development use only!")
			}

		case "terminate":
			switch val {
			case "1":
				terminateMode = true
				logrus.Warn("Using terminate mode in Windows exec driver. This is for testing purposes only.")
			}

		default:
			return nil, fmt.Errorf("Unrecognised exec driver option %s\n", key)
		}
	}

	return &driver{
		root:             root,
		initPath:         initPath,
		activeContainers: make(map[string]*activeContainer),
	}, nil
}

// setupEnvironmentVariables convert a string array of environment variables
// into a map as required by the HCS. Source array is in format [v1=k1] [v2=k2] etc.
func setupEnvironmentVariables(a []string) map[string]string {
	r := make(map[string]string)
	for _, s := range a {
		arr := strings.Split(s, "=")
		if len(arr) == 2 {
			r[arr[0]] = arr[1]
		}
	}
	return r
}
