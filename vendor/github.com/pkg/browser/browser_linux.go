package browser

import (
	"os/exec"
	"strings"
)

func openBrowser(url string) error {
	providers := []string{"xdg-open", "x-www-browser", "www-browser"}

	// There are multiple possible providers to open a browser on linux
	// One of them is xdg-open, another is x-www-browser, then there's www-browser, etc.
	// Look for one that exists and run it
	for _, provider := range providers {
		if _, err := exec.LookPath(provider); err == nil {
			return runCmd(provider, url)
		}
	}

	return &exec.Error{Name: strings.Join(providers, ","), Err: exec.ErrNotFound}
}

func setFlags(cmd *exec.Cmd) {}
