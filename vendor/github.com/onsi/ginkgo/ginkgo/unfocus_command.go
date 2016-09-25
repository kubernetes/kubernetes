package main

import (
	"flag"
	"fmt"
	"os/exec"
)

func BuildUnfocusCommand() *Command {
	return &Command{
		Name:         "unfocus",
		AltName:      "blur",
		FlagSet:      flag.NewFlagSet("unfocus", flag.ExitOnError),
		UsageCommand: "ginkgo unfocus (or ginkgo blur)",
		Usage: []string{
			"Recursively unfocuses any focused tests under the current directory",
		},
		Command: unfocusSpecs,
	}
}

func unfocusSpecs([]string, []string) {
	unfocus("Describe")
	unfocus("Context")
	unfocus("It")
	unfocus("Measure")
	unfocus("DescribeTable")
	unfocus("Entry")
}

func unfocus(component string) {
	fmt.Printf("Removing F%s...\n", component)
	cmd := exec.Command("gofmt", fmt.Sprintf("-r=F%s -> %s", component, component), "-w", ".")
	out, _ := cmd.CombinedOutput()
	if string(out) != "" {
		println(string(out))
	}
}
