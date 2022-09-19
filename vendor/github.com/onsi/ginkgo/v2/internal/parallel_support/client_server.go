package parallel_support

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

type BeforeSuiteState struct {
	Data  []byte
	State types.SpecState
}

type ParallelIndexCounter struct {
	Index int
}

var ErrorGone = fmt.Errorf("gone")
var ErrorFailed = fmt.Errorf("failed")
var ErrorEarly = fmt.Errorf("early")

var POLLING_INTERVAL = 50 * time.Millisecond

type Server interface {
	Start()
	Close()
	Address() string
	RegisterAlive(node int, alive func() bool)
	GetSuiteDone() chan interface{}
	GetOutputDestination() io.Writer
	SetOutputDestination(io.Writer)
}

type Client interface {
	Connect() bool
	Close() error

	PostSuiteWillBegin(report types.Report) error
	PostDidRun(report types.SpecReport) error
	PostSuiteDidEnd(report types.Report) error
	PostSynchronizedBeforeSuiteCompleted(state types.SpecState, data []byte) error
	BlockUntilSynchronizedBeforeSuiteData() (types.SpecState, []byte, error)
	BlockUntilNonprimaryProcsHaveFinished() error
	BlockUntilAggregatedNonprimaryProcsReport() (types.Report, error)
	FetchNextCounter() (int, error)
	PostAbort() error
	ShouldAbort() bool
	PostEmitProgressReport(report types.ProgressReport) error
	Write(p []byte) (int, error)
}

func NewServer(parallelTotal int, reporter reporters.Reporter) (Server, error) {
	if os.Getenv("GINKGO_PARALLEL_PROTOCOL") == "HTTP" {
		return newHttpServer(parallelTotal, reporter)
	} else {
		return newRPCServer(parallelTotal, reporter)
	}
}

func NewClient(serverHost string) Client {
	if os.Getenv("GINKGO_PARALLEL_PROTOCOL") == "HTTP" {
		return newHttpClient(serverHost)
	} else {
		return newRPCClient(serverHost)
	}
}
