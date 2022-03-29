package internal

import (
	"bytes"
	"io"
	"os"
	"time"
)

const BAILOUT_TIME = 1 * time.Second
const BAILOUT_MESSAGE = `Ginkgo detected an issue while intercepting output.

When running in parallel, Ginkgo captures stdout and stderr output
and attaches it to the running spec.  It looks like that process is getting
stuck for this suite.

This usually happens if you, or a library you are using, spin up an external
process and set cmd.Stdout = os.Stdout and/or cmd.Stderr = os.Stderr.  This
causes the external process to keep Ginkgo's output interceptor pipe open and
causes output interception to hang.

Ginkgo has detected this and shortcircuited the capture process.  The specs
will continue running after this message however output from the external
process that caused this issue will not be captured.

You have several options to fix this.  In preferred order they are:

1. Pass GinkgoWriter instead of os.Stdout or os.Stderr to your process.
2. Ensure your process exits before the current spec completes.  If your
process is long-lived and must cross spec boundaries, this option won't
work for you.
3. Pause Ginkgo's output interceptor before starting your process and then
resume it after.  Use PauseOutputInterception() and ResumeOutputInterception()
to do this.
4. Set --output-interceptor-mode=none when running your Ginkgo suite.  This will
turn off all output interception but allow specs to run in parallel without this
issue.  You may miss important output if you do this including output from Go's
race detector.

More details on issue #851 - https://github.com/onsi/ginkgo/issues/851
`

/*
The OutputInterceptor is used by to
intercept and capture all stdin and stderr output during a test run.
*/
type OutputInterceptor interface {
	StartInterceptingOutput()
	StartInterceptingOutputAndForwardTo(io.Writer)
	StopInterceptingAndReturnOutput() string

	PauseIntercepting()
	ResumeIntercepting()

	Shutdown()
}

type NoopOutputInterceptor struct{}

func (interceptor NoopOutputInterceptor) StartInterceptingOutput()                      {}
func (interceptor NoopOutputInterceptor) StartInterceptingOutputAndForwardTo(io.Writer) {}
func (interceptor NoopOutputInterceptor) StopInterceptingAndReturnOutput() string       { return "" }
func (interceptor NoopOutputInterceptor) PauseIntercepting()                            {}
func (interceptor NoopOutputInterceptor) ResumeIntercepting()                           {}
func (interceptor NoopOutputInterceptor) Shutdown()                                     {}

type pipePair struct {
	reader *os.File
	writer *os.File
}

func startPipeFactory(pipeChannel chan pipePair, shutdown chan interface{}) {
	for {
		//make the next pipe...
		pair := pipePair{}
		pair.reader, pair.writer, _ = os.Pipe()
		select {
		//...and provide it to the next consumer (they are responsible for closing the files)
		case pipeChannel <- pair:
			continue
		//...or close the files if we were told to shutdown
		case <-shutdown:
			pair.reader.Close()
			pair.writer.Close()
			return
		}
	}
}

type interceptorImplementation interface {
	CreateStdoutStderrClones() (*os.File, *os.File)
	ConnectPipeToStdoutStderr(*os.File)
	RestoreStdoutStderrFromClones(*os.File, *os.File)
	ShutdownClones(*os.File, *os.File)
}

type genericOutputInterceptor struct {
	intercepting bool

	stdoutClone *os.File
	stderrClone *os.File
	pipe        pipePair

	shutdown           chan interface{}
	emergencyBailout   chan interface{}
	pipeChannel        chan pipePair
	interceptedContent chan string

	forwardTo         io.Writer
	accumulatedOutput string

	implementation interceptorImplementation
}

func (interceptor *genericOutputInterceptor) StartInterceptingOutput() {
	interceptor.StartInterceptingOutputAndForwardTo(io.Discard)
}

func (interceptor *genericOutputInterceptor) StartInterceptingOutputAndForwardTo(w io.Writer) {
	if interceptor.intercepting {
		return
	}
	interceptor.accumulatedOutput = ""
	interceptor.forwardTo = w
	interceptor.ResumeIntercepting()
}

func (interceptor *genericOutputInterceptor) StopInterceptingAndReturnOutput() string {
	if interceptor.intercepting {
		interceptor.PauseIntercepting()
	}
	return interceptor.accumulatedOutput
}

func (interceptor *genericOutputInterceptor) ResumeIntercepting() {
	if interceptor.intercepting {
		return
	}
	interceptor.intercepting = true
	if interceptor.stdoutClone == nil {
		interceptor.stdoutClone, interceptor.stderrClone = interceptor.implementation.CreateStdoutStderrClones()
		interceptor.shutdown = make(chan interface{})
		go startPipeFactory(interceptor.pipeChannel, interceptor.shutdown)
	}

	// Now we make a pipe, we'll use this to redirect the input to the 1 and 2 file descriptors (this is how everything else in the world is tring to log to stdout and stderr)
	// we get the pipe from our pipe factory.  it runs in the background so we can request the next pipe while the spec being intercepted is running
	interceptor.pipe = <-interceptor.pipeChannel

	interceptor.emergencyBailout = make(chan interface{})

	//Spin up a goroutine to copy data from the pipe into a buffer, this is how we capture any output the user is emitting
	go func() {
		buffer := &bytes.Buffer{}
		destination := io.MultiWriter(buffer, interceptor.forwardTo)
		copyFinished := make(chan interface{})
		reader := interceptor.pipe.reader
		go func() {
			io.Copy(destination, reader)
			reader.Close() // close the read end of the pipe so we don't leak a file descriptor
			close(copyFinished)
		}()
		select {
		case <-copyFinished:
			interceptor.interceptedContent <- buffer.String()
		case <-interceptor.emergencyBailout:
			interceptor.interceptedContent <- ""
		}
	}()

	interceptor.implementation.ConnectPipeToStdoutStderr(interceptor.pipe.writer)
}

func (interceptor *genericOutputInterceptor) PauseIntercepting() {
	if !interceptor.intercepting {
		return
	}
	// first we have to close the write end of the pipe.  To do this we have to close all file descriptors pointing
	// to the write end.  So that would be the pipewriter itself, and FD #1 and FD #2 if we've Dup2'd them
	interceptor.pipe.writer.Close() // the pipewriter itself

	// we also need to stop intercepting. we do that by reconnecting the stdout and stderr file descriptions back to their respective #1 and #2 file descriptors;
	// this also closes #1 and #2 before it points that their original stdout and stderr file descriptions
	interceptor.implementation.RestoreStdoutStderrFromClones(interceptor.stdoutClone, interceptor.stderrClone)

	var content string
	select {
	case content = <-interceptor.interceptedContent:
	case <-time.After(BAILOUT_TIME):
		/*
			By closing all the pipe writer's file descriptors associated with the pipe writer's file description the io.Copy reading from the reader
			should eventually receive an EOF and exit.

			**However**, if the user has spun up an external process and passed in os.Stdout/os.Stderr to cmd.Stdout/cmd.Stderr then the external process
			will have a file descriptor pointing to the pipe writer's file description and it will not close until the external process exits.

			That would leave us hanging here waiting for the io.Copy to close forever.  Instead we invoke this emergency escape valve.  This returns whatever
			content we've got but leaves the io.Copy running.  This ensures the external process can continue writing without hanging at the cost of leaking a goroutine
			and file descriptor (those these will be cleaned up when the process exits).

			We tack on a message to notify the user that they've hit this edgecase and encourage them to address it.
		*/
		close(interceptor.emergencyBailout)
		content = <-interceptor.interceptedContent + BAILOUT_MESSAGE
	}

	interceptor.accumulatedOutput += content
	interceptor.intercepting = false
}

func (interceptor *genericOutputInterceptor) Shutdown() {
	interceptor.PauseIntercepting()

	if interceptor.stdoutClone != nil {
		close(interceptor.shutdown)
		interceptor.implementation.ShutdownClones(interceptor.stdoutClone, interceptor.stderrClone)
		interceptor.stdoutClone = nil
		interceptor.stderrClone = nil
	}
}

/* This is used on windows builds but included here so it can be explicitly tested on unix systems too */
func NewOSGlobalReassigningOutputInterceptor() OutputInterceptor {
	return &genericOutputInterceptor{
		interceptedContent: make(chan string),
		pipeChannel:        make(chan pipePair),
		shutdown:           make(chan interface{}),
		implementation:     &osGlobalReassigningOutputInterceptorImpl{},
	}
}

type osGlobalReassigningOutputInterceptorImpl struct{}

func (impl *osGlobalReassigningOutputInterceptorImpl) CreateStdoutStderrClones() (*os.File, *os.File) {
	return os.Stdout, os.Stderr
}

func (impl *osGlobalReassigningOutputInterceptorImpl) ConnectPipeToStdoutStderr(pipeWriter *os.File) {
	os.Stdout = pipeWriter
	os.Stderr = pipeWriter
}

func (impl *osGlobalReassigningOutputInterceptorImpl) RestoreStdoutStderrFromClones(stdoutClone *os.File, stderrClone *os.File) {
	os.Stdout = stdoutClone
	os.Stderr = stderrClone
}

func (impl *osGlobalReassigningOutputInterceptorImpl) ShutdownClones(_ *os.File, _ *os.File) {
	//noop
}
