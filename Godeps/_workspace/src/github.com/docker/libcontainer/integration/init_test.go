package integration

import (
	"encoding/json"
	"log"
	"os"
	"runtime"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/namespaces"
	_ "github.com/docker/libcontainer/namespaces/nsenter"
)

// init runs the libcontainer initialization code because of the busybox style needs
// to work around the go runtime and the issues with forking
func init() {
	if len(os.Args) < 2 {
		return
	}
	// handle init
	if len(os.Args) >= 2 && os.Args[1] == "init" {
		runtime.LockOSThread()

		container, err := loadConfig()
		if err != nil {
			log.Fatal(err)
		}

		rootfs, err := os.Getwd()
		if err != nil {
			log.Fatal(err)
		}

		if err := namespaces.Init(container, rootfs, "", os.NewFile(3, "pipe"), os.Args[3:]); err != nil {
			log.Fatalf("unable to initialize for container: %s", err)
		}
		os.Exit(1)
	}

	// handle execin
	if len(os.Args) >= 2 && os.Args[0] == "nsenter-exec" {
		runtime.LockOSThread()

		// User args are passed after '--' in the command line.
		userArgs := findUserArgs()

		config, err := loadConfigFromFd()
		if err != nil {
			log.Fatalf("docker-exec: unable to receive config from sync pipe: %s", err)
		}

		if err := namespaces.FinalizeSetns(config, userArgs); err != nil {
			log.Fatalf("docker-exec: failed to exec: %s", err)
		}
		os.Exit(1)
	}
}

func findUserArgs() []string {
	for i, a := range os.Args {
		if a == "--" {
			return os.Args[i+1:]
		}
	}
	return []string{}
}

// loadConfigFromFd loads a container's config from the sync pipe that is provided by
// fd 3 when running a process
func loadConfigFromFd() (*libcontainer.Config, error) {
	var config *libcontainer.Config
	if err := json.NewDecoder(os.NewFile(3, "child")).Decode(&config); err != nil {
		return nil, err
	}
	return config, nil
}
