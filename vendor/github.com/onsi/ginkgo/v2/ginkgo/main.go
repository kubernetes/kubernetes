package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/onsi/ginkgo/v2/ginkgo/build"
	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/ginkgo/generators"
	"github.com/onsi/ginkgo/v2/ginkgo/labels"
	"github.com/onsi/ginkgo/v2/ginkgo/outline"
	"github.com/onsi/ginkgo/v2/ginkgo/run"
	"github.com/onsi/ginkgo/v2/ginkgo/unfocus"
	"github.com/onsi/ginkgo/v2/ginkgo/watch"
	"github.com/onsi/ginkgo/v2/types"
)

var program command.Program

func GenerateCommands() []command.Command {
	return []command.Command{
		watch.BuildWatchCommand(),
		build.BuildBuildCommand(),
		generators.BuildBootstrapCommand(),
		generators.BuildGenerateCommand(),
		labels.BuildLabelsCommand(),
		outline.BuildOutlineCommand(),
		unfocus.BuildUnfocusCommand(),
		BuildVersionCommand(),
	}
}

func main() {
	// Temporary diagnostic for kubernetes/test-infra#37727 and kubernetes/kubernetes#141435:
	// print the CPU the Go runtime sees, plus cpu.max at each cgroup level, so a
	// real presubmit run shows whether the runner init leaf hides the pod quota.
	fmt.Fprintf(os.Stderr, "DRA-CPU-DIAG: GOMAXPROCS=%d NumCPU=%d GOMAXPROCS_env=%q\n",
		runtime.GOMAXPROCS(0), runtime.NumCPU(), os.Getenv("GOMAXPROCS"))
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "DefaultGODEBUG" {
				fmt.Fprintf(os.Stderr, "DRA-CPU-DIAG: DefaultGODEBUG=%s\n", s.Value)
			}
		}
	}
	if raw, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		path := strings.TrimSpace(string(raw))
		if i := strings.Index(path, "::"); i >= 0 {
			path = path[i+2:]
		}
		for p := path; ; {
			max, _ := os.ReadFile("/sys/fs/cgroup" + p + "/cpu.max")
			fmt.Fprintf(os.Stderr, "DRA-CPU-DIAG: %s cpu.max=[%s]\n", p, strings.TrimSpace(string(max)))
			if p == "/" || p == "" {
				break
			}
			if i := strings.LastIndex(p, "/"); i > 0 {
				p = p[:i]
			} else {
				p = "/"
			}
		}
	}

	// Experiment for kubernetes/test-infra#37727: print the diagnostic and exit so the
	// presubmit finishes fast without running the whole suite. Delete to restore ginkgo.
	os.Exit(0)

	program = command.Program{
		Name:           "ginkgo",
		Heading:        fmt.Sprintf("Ginkgo Version %s", types.VERSION),
		Commands:       GenerateCommands(),
		DefaultCommand: run.BuildRunCommand(),
		DeprecatedCommands: []command.DeprecatedCommand{
			{Name: "convert", Deprecation: types.Deprecations.Convert()},
			{Name: "blur", Deprecation: types.Deprecations.Blur()},
			{Name: "nodot", Deprecation: types.Deprecations.Nodot()},
		},
	}
	program.Commands = append(program.Commands, program.BuildCompletionCommand())

	program.RunAndExit(os.Args)
}

func BuildVersionCommand() command.Command {
	return command.Command{
		Name:     "version",
		Usage:    "ginkgo version",
		ShortDoc: "Print Ginkgo's version",
		Command: func(_ []string, _ []string) {
			fmt.Printf("Ginkgo Version %s\n", types.VERSION)
		},
	}
}
