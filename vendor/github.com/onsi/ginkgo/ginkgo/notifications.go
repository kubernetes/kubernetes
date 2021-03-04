package main

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"

	"github.com/onsi/ginkgo/config"
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
		onLinux := (runtime.GOOS == "linux")
		onOSX := (runtime.GOOS == "darwin")
		if onOSX {

			_, err := exec.LookPath("terminal-notifier")
			if err != nil {
				fmt.Printf(`--notify requires terminal-notifier, which you don't seem to have installed.

OSX:

To remedy this:

    brew install terminal-notifier

To learn more about terminal-notifier:

    https://github.com/alloy/terminal-notifier
`)
				os.Exit(1)
			}

		} else if onLinux {

			_, err := exec.LookPath("notify-send")
			if err != nil {
				fmt.Printf(`--notify requires terminal-notifier or notify-send, which you don't seem to have installed.

Linux:

Download and install notify-send for your distribution
`)
				os.Exit(1)
			}

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

	if n.commandFlags.Notify {
		onLinux := (runtime.GOOS == "linux")
		onOSX := (runtime.GOOS == "darwin")

		if onOSX {

			_, err := exec.LookPath("terminal-notifier")
			if err == nil {
				args := []string{"-title", title, "-subtitle", subtitle, "-group", "com.onsi.ginkgo"}
				terminal := os.Getenv("TERM_PROGRAM")
				if terminal == "iTerm.app" {
					args = append(args, "-activate", "com.googlecode.iterm2")
				} else if terminal == "Apple_Terminal" {
					args = append(args, "-activate", "com.apple.Terminal")
				}

				exec.Command("terminal-notifier", args...).Run()
			}

		} else if onLinux {

			_, err := exec.LookPath("notify-send")
			if err == nil {
				args := []string{"-a", "ginkgo", title, subtitle}
				exec.Command("notify-send", args...).Run()
			}

		}
	}
}

func (n *Notifier) RunCommand(suite testsuite.TestSuite, suitePassed bool) {

	command := n.commandFlags.AfterSuiteHook
	if command != "" {

		// Allow for string replacement to pass input to the command
		passed := "[FAIL]"
		if suitePassed {
			passed = "[PASS]"
		}
		command = strings.Replace(command, "(ginkgo-suite-passed)", passed, -1)
		command = strings.Replace(command, "(ginkgo-suite-name)", suite.PackageName, -1)

		// Must break command into parts
		splitArgs := regexp.MustCompile(`'.+'|".+"|\S+`)
		parts := splitArgs.FindAllString(command, -1)

		output, err := exec.Command(parts[0], parts[1:]...).CombinedOutput()
		if err != nil {
			fmt.Println("Post-suite command failed:")
			if config.DefaultReporterConfig.NoColor {
				fmt.Printf("\t%s\n", output)
			} else {
				fmt.Printf("\t%s%s%s\n", redColor, string(output), defaultStyle)
			}
			n.SendNotification("Ginkgo [ERROR]", fmt.Sprintf(`After suite command "%s" failed`, n.commandFlags.AfterSuiteHook))
		} else {
			fmt.Println("Post-suite command succeeded:")
			if config.DefaultReporterConfig.NoColor {
				fmt.Printf("\t%s\n", output)
			} else {
				fmt.Printf("\t%s%s%s\n", greenColor, string(output), defaultStyle)
			}
		}
	}
}
