package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os/exec"
	"strings"
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
	unfocus("Specify")
	unfocus("When")
}

func unfocus(component string) {
	fmt.Printf("Removing F%s...\n", component)
	files, err := ioutil.ReadDir(".")
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	for _, f := range files {
		// Exclude "vendor" directory
		if f.IsDir() && f.Name() == "vendor" {
			continue
		}
		// Exclude non-go files in the current directory
		if !f.IsDir() && !strings.HasSuffix(f.Name(), ".go") {
			continue
		}
		// Recursively run `gofmt` otherwise
		cmd := exec.Command("gofmt", fmt.Sprintf("-r=F%s -> %s", component, component), "-w", f.Name())
		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Println(err.Error())
		}
		if string(out) != "" {
			fmt.Println(string(out))
		}
	}
}
