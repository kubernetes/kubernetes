package bidichk

import (
	"fmt"
	"os"
)

var Version = "bidichk version dev"

type versionFlag struct{}

func (versionFlag) IsBoolFlag() bool { return true }
func (versionFlag) Get() interface{} { return nil }
func (versionFlag) String() string   { return "" }
func (versionFlag) Set(s string) error {
	fmt.Println(Version)
	os.Exit(0)
	return nil
}
