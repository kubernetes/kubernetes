package global

import (
	"github.com/onsi/ginkgo/v2/internal"
)

var Suite *internal.Suite
var Failer *internal.Failer
var backupSuite *internal.Suite

// SuiteDidRun tracks whether RunSpecs has already been invoked for the current global
// suite. It lives here (rather than in package ginkgo) so that InitializeGlobals can
// clear it, allowing extensions/globals.Reset to support running multiple suites
// sequentially in a single process.
var SuiteDidRun bool

func init() {
	InitializeGlobals()
}

func InitializeGlobals() {
	Failer = internal.NewFailer()
	Suite = internal.NewSuite()
	SuiteDidRun = false
}

func PushClone() error {
	var err error
	backupSuite, err = Suite.Clone()
	return err
}

func PopClone() {
	Suite = backupSuite
}
