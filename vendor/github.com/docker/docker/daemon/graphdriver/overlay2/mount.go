// +build linux

package overlay2

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/docker/docker/pkg/reexec"
	"golang.org/x/sys/unix"
)

func init() {
	reexec.Register("docker-mountfrom", mountFromMain)
}

func fatal(err error) {
	fmt.Fprint(os.Stderr, err)
	os.Exit(1)
}

type mountOptions struct {
	Device string
	Target string
	Type   string
	Label  string
	Flag   uint32
}

func mountFrom(dir, device, target, mType string, flags uintptr, label string) error {
	options := &mountOptions{
		Device: device,
		Target: target,
		Type:   mType,
		Flag:   uint32(flags),
		Label:  label,
	}

	cmd := reexec.Command("docker-mountfrom", dir)
	w, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("mountfrom error on pipe creation: %v", err)
	}

	output := bytes.NewBuffer(nil)
	cmd.Stdout = output
	cmd.Stderr = output

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("mountfrom error on re-exec cmd: %v", err)
	}
	//write the options to the pipe for the untar exec to read
	if err := json.NewEncoder(w).Encode(options); err != nil {
		return fmt.Errorf("mountfrom json encode to pipe failed: %v", err)
	}
	w.Close()

	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("mountfrom re-exec error: %v: output: %s", err, output)
	}
	return nil
}

// mountfromMain is the entry-point for docker-mountfrom on re-exec.
func mountFromMain() {
	runtime.LockOSThread()
	flag.Parse()

	var options *mountOptions

	if err := json.NewDecoder(os.Stdin).Decode(&options); err != nil {
		fatal(err)
	}

	if err := os.Chdir(flag.Arg(0)); err != nil {
		fatal(err)
	}

	if err := unix.Mount(options.Device, options.Target, options.Type, uintptr(options.Flag), options.Label); err != nil {
		fatal(err)
	}

	os.Exit(0)
}
