package invoke

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/appc/cni/pkg/types"
)

func pluginErr(err error, output []byte) error {
	if _, ok := err.(*exec.ExitError); ok {
		emsg := types.Error{}
		if perr := json.Unmarshal(output, &emsg); perr != nil {
			return fmt.Errorf("netplugin failed but error parsing its diagnostic message %q: %v", string(output), perr)
		}
		details := ""
		if emsg.Details != "" {
			details = fmt.Sprintf("; %v", emsg.Details)
		}
		return fmt.Errorf("%v%v", emsg.Msg, details)
	}

	return err
}

// ExecAdd executes IPAM plugin, assuming CNI_COMMAND == ADD.
// Parses and returns resulting IPConfig
func ExecPlugin(pluginPath string, netconf []byte, env []string) (*types.Result, error) {
	if pluginPath == "" {
		return nil, fmt.Errorf("could not find %q plugin", filepath.Base(pluginPath))
	}

	stdout := &bytes.Buffer{}

	c := exec.Cmd{
		Env:    env,
		Path:   pluginPath,
		Args:   []string{pluginPath},
		Stdin:  bytes.NewBuffer(netconf),
		Stdout: stdout,
		Stderr: os.Stderr,
	}
	if err := c.Run(); err != nil {
		return nil, pluginErr(err, stdout.Bytes())
	}

	res := &types.Result{}
	err := json.Unmarshal(stdout.Bytes(), res)
	return res, err
}
