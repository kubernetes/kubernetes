package browser

import "os/exec"

func openBrowser(url string) error {
	return runCmd("open", url)
}

func setFlags(cmd *exec.Cmd) {}
