// +build !windows

package idtools

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

func resolveBinary(binname string) (string, error) {
	binaryPath, err := exec.LookPath(binname)
	if err != nil {
		return "", err
	}
	resolvedPath, err := filepath.EvalSymlinks(binaryPath)
	if err != nil {
		return "", err
	}
	//only return no error if the final resolved binary basename
	//matches what was searched for
	if filepath.Base(resolvedPath) == binname {
		return resolvedPath, nil
	}
	return "", fmt.Errorf("Binary %q does not resolve to a binary of that name in $PATH (%q)", binname, resolvedPath)
}

func execCmd(cmd, args string) ([]byte, error) {
	execCmd := exec.Command(cmd, strings.Split(args, " ")...)
	return execCmd.CombinedOutput()
}
