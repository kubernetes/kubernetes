// +build windows

package open

import (
	"os/exec"
	"strings"
)

func cleaninput(input string) string {
	r := strings.NewReplacer("&", "^&")
	return r.Replace(input)
}

func open(input string) *exec.Cmd {
	return exec.Command("cmd", "/C", "start", "", cleaninput(input))
}

func openWith(input string, appName string) *exec.Cmd {
	return exec.Command("cmd", "/C", "start", "", appName, cleaninput(input))
}
