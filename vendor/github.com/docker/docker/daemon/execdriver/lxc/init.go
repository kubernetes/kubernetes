// +build linux

package lxc

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/reexec"
)

// Args provided to the init function for a driver
type InitArgs struct {
	User       string
	Gateway    string
	Ip         string
	WorkDir    string
	Privileged bool
	Env        []string
	Args       []string
	Mtu        int
	Console    string
	Pipe       int
	Root       string
	CapAdd     string
	CapDrop    string
}

func init() {
	// like always lxc requires a hack to get this to work
	reexec.Register("/.dockerinit", dockerInititalizer)
}

func dockerInititalizer() {
	initializer()
}

// initializer is the lxc driver's init function that is run inside the namespace to setup
// additional configurations
func initializer() {
	runtime.LockOSThread()

	args := getArgs()

	if err := setupNamespace(args); err != nil {
		logrus.Fatal(err)
	}
}

func setupNamespace(args *InitArgs) error {
	if err := setupEnv(args); err != nil {
		return err
	}

	if err := finalizeNamespace(args); err != nil {
		return err
	}

	path, err := exec.LookPath(args.Args[0])
	if err != nil {
		logrus.Infof("Unable to locate %v", args.Args[0])
		os.Exit(127)
	}

	if err := syscall.Exec(path, args.Args, os.Environ()); err != nil {
		return fmt.Errorf("dockerinit unable to execute %s - %s", path, err)
	}

	return nil
}

func getArgs() *InitArgs {
	var (
		// Get cmdline arguments
		user       = flag.String("u", "", "username or uid")
		gateway    = flag.String("g", "", "gateway address")
		ip         = flag.String("i", "", "ip address")
		workDir    = flag.String("w", "", "workdir")
		privileged = flag.Bool("privileged", false, "privileged mode")
		mtu        = flag.Int("mtu", 1500, "interface mtu")
		capAdd     = flag.String("cap-add", "", "capabilities to add")
		capDrop    = flag.String("cap-drop", "", "capabilities to drop")
	)

	flag.Parse()

	return &InitArgs{
		User:       *user,
		Gateway:    *gateway,
		Ip:         *ip,
		WorkDir:    *workDir,
		Privileged: *privileged,
		Args:       flag.Args(),
		Mtu:        *mtu,
		CapAdd:     *capAdd,
		CapDrop:    *capDrop,
	}
}

// Clear environment pollution introduced by lxc-start
func setupEnv(args *InitArgs) error {
	// Get env
	var env []string
	dockerenv, err := os.Open(".dockerenv")
	if err != nil {
		return fmt.Errorf("Unable to load environment variables: %v", err)
	}
	defer dockerenv.Close()
	if err := json.NewDecoder(dockerenv).Decode(&env); err != nil {
		return fmt.Errorf("Unable to decode environment variables: %v", err)
	}
	// Propagate the plugin-specific container env variable
	env = append(env, "container="+os.Getenv("container"))

	args.Env = env

	os.Clearenv()
	for _, kv := range args.Env {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 1 {
			parts = append(parts, "")
		}
		os.Setenv(parts[0], parts[1])
	}

	return nil
}

// Setup working directory
func setupWorkingDirectory(args *InitArgs) error {
	if args.WorkDir == "" {
		return nil
	}
	if err := syscall.Chdir(args.WorkDir); err != nil {
		return fmt.Errorf("Unable to change dir to %v: %v", args.WorkDir, err)
	}
	return nil
}
