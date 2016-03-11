package browser

import (
	"strings"
)

func openBrowser(url string) error {
	r := strings.NewReplacer("&", "^&")
	return runCmd("cmd", "/c", "start", r.Replace(url))
}
