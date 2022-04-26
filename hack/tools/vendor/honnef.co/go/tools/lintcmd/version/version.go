package version

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

const Version = "2021.1.2"
const MachineVersion = "v0.2.2"

// version returns a version descriptor and reports whether the
// version is a known release.
func version(human, machine string) (human_, machine_ string, known bool) {
	if human != "devel" {
		return human, machine, true
	}
	v, ok := buildInfoVersion()
	if ok {
		return v, "", false
	}
	return "devel", "", false
}

func Print(human, machine string) {
	human, machine, release := version(human, machine)

	if release {
		fmt.Printf("%s %s (%s)\n", filepath.Base(os.Args[0]), human, machine)
	} else if human == "devel" {
		fmt.Printf("%s (no version)\n", filepath.Base(os.Args[0]))
	} else {
		fmt.Printf("%s (devel, %s)\n", filepath.Base(os.Args[0]), human)
	}
}

func Verbose(human, machine string) {
	Print(human, machine)
	fmt.Println()
	fmt.Println("Compiled with Go version:", runtime.Version())
	printBuildInfo()
}
