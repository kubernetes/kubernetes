//go:generate mkwinsyscall -output zbrowser_windows.go browser_windows.go
//sys ShellExecute(hwnd int, verb string, file string, args string, cwd string, showCmd int) (err error) = shell32.ShellExecuteW
package browser

import "os/exec"
const SW_SHOWNORMAL = 1

func openBrowser(url string) error {
   return ShellExecute(0, "", url, "", "", SW_SHOWNORMAL)
}

func setFlags(cmd *exec.Cmd) {
}
