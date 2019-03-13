package main

import (
	"fmt"
	"log"
	"runtime"
	"strconv"
	"strings"
)

const version = "80-k8s-r1"

var cmdVersion = &Command{
	Name:  "version",
	Short: "show version info",
	Long: `

Displays the version of godep as well as the target OS, architecture and go runtime version.
`,
	Run: runVersion,
}

func versionString() string {
	return fmt.Sprintf("godep v%s (%s/%s/%s)", version, runtime.GOOS, runtime.GOARCH, runtime.Version())
}

func runVersion(cmd *Command, args []string) {
	fmt.Printf("%s\n", versionString())
}

func GoVersionFields(c rune) bool {
	return c == 'g' || c == 'o' || c == '.'
}

// isSameOrNewer go version (goA.B)
// go1.6 >= go1.6 == true
// go1.5 >= go1.6 == false
func isSameOrNewer(base, check string) bool {
	if base == check {
		return true
	}
	if strings.HasPrefix(check, "devel-") {
		return true
	}
	bp := strings.FieldsFunc(base, GoVersionFields)
	cp := strings.FieldsFunc(check, GoVersionFields)
	if len(bp) < 2 || len(cp) < 2 {
		log.Fatalf("Error comparing %s to %s\n", base, check)
	}
	if bp[0] == cp[0] { // We only have go version 1 right now
		bm, err := strconv.Atoi(bp[1])
		// These errors are unlikely and there is nothing nice to do here anyway
		if err != nil {
			panic(err)
		}
		cm, err := strconv.Atoi(cp[1])
		if err != nil {
			panic(err)
		}
		return cm >= bm
	}
	return false
}
