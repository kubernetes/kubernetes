package main

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/onsi/ginkgo/ginkgo/testsuite"
)

type Notifier struct {
	commandFlags *RunWatchAndBuildCommandFlags
}

func NewNotifier(commandFlags *RunWatchAndBuildCommandFlags) *Notifier {
	return &Notifier{
		commandFlags: commandFlags,
	}
}

func (n *Notifier) VerifyNotificationsAreAvailable() {
	if n.commandFlags.Notify {
		_, err := exec.LookPath("terminal-notifier")
		if err != nil {
			fmt.Printf(`--notify requires terminal-notifier, which you don't seem to have installed.

To remedy this:

    brew install terminal-notifier

To learn more about terminal-notifier:

    https://github.com/alloy/terminal-notifier
`)
			os.Exit(1)
		}
	}
}

func (n *Notifier) SendSuiteCompletionNotification(suite testsuite.TestSuite, suitePassed bool) {
	if suitePassed {
		n.SendNotification("Ginkgo [PASS]", fmt.Sprintf(`Test suite for "%s" passed.`, suite.PackageName))
	} else {
		n.SendNotification("Ginkgo [FAIL]", fmt.Sprintf(`Test suite for "%s" failed.`, suite.PackageName))
	}
}

func (n *Notifier) SendNotification(title string, subtitle string) {
	args := []string{"-title", title, "-subtitle", subtitle, "-group", "com.onsi.ginkgo"}

	terminal := os.Getenv("TERM_PROGRAM")
	if terminal == "iTerm.app" {
		args = append(args, "-activate", "com.googlecode.iterm2")
	} else if terminal == "Apple_Terminal" {
		args = append(args, "-activate", "com.apple.Terminal")
	}

	if n.commandFlags.Notify {
		exec.Command("terminal-notifier", args...).Run()
	}
}
