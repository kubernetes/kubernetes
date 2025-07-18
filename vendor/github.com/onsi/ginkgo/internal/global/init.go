package global

import (
	"time"

	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/internal/suite"
)

const DefaultTimeout = time.Duration(1 * time.Second)

var Suite *suite.Suite
var Failer *failer.Failer

func init() {
	InitializeGlobals()
}

func InitializeGlobals() {
	Failer = failer.New()
	Suite = suite.New(Failer)
}
