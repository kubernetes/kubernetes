/*
Package processcreds is a credential Provider to retrieve `credential_process`
credentials.

WARNING: The following describes a method of sourcing credentials from an external
process. This can potentially be dangerous, so proceed with caution. Other
credential providers should be preferred if at all possible. If using this
option, you should make sure that the config file is as locked down as possible
using security best practices for your operating system.

You can use credentials from a `credential_process` in a variety of ways.

One way is to setup your shared config file, located in the default
location, with the `credential_process` key and the command you want to be
called. You also need to set the AWS_SDK_LOAD_CONFIG environment variable
(e.g., `export AWS_SDK_LOAD_CONFIG=1`) to use the shared config file.

    [default]
    credential_process = /command/to/call

Creating a new session will use the credential process to retrieve credentials.
NOTE: If there are credentials in the profile you are using, the credential
process will not be used.

    // Initialize a session to load credentials.
    sess, _ := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1")},
    )

    // Create S3 service client to use the credentials.
    svc := s3.New(sess)

Another way to use the `credential_process` method is by using
`credentials.NewCredentials()` and providing a command to be executed to
retrieve credentials:

    // Create credentials using the ProcessProvider.
    creds := processcreds.NewCredentials("/path/to/command")

    // Create service client value configured for credentials.
    svc := s3.New(sess, &aws.Config{Credentials: creds})

You can set a non-default timeout for the `credential_process` with another
constructor, `credentials.NewCredentialsTimeout()`, providing the timeout. To
set a one minute timeout:

    // Create credentials using the ProcessProvider.
    creds := processcreds.NewCredentialsTimeout(
        "/path/to/command",
        time.Duration(500) * time.Millisecond)

If you need more control, you can set any configurable options in the
credentials using one or more option functions. For example, you can set a two
minute timeout, a credential duration of 60 minutes, and a maximum stdout
buffer size of 2k.

    creds := processcreds.NewCredentials(
        "/path/to/command",
        func(opt *ProcessProvider) {
            opt.Timeout = time.Duration(2) * time.Minute
            opt.Duration = time.Duration(60) * time.Minute
            opt.MaxBufSize = 2048
        })

You can also use your own `exec.Cmd`:

	// Create an exec.Cmd
	myCommand := exec.Command("/path/to/command")

	// Create credentials using your exec.Cmd and custom timeout
	creds := processcreds.NewCredentialsCommand(
		myCommand,
		func(opt *processcreds.ProcessProvider) {
			opt.Timeout = time.Duration(1) * time.Second
		})
*/
package processcreds

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
)

const (
	// ProviderName is the name this credentials provider will label any
	// returned credentials Value with.
	ProviderName = `ProcessProvider`

	// ErrCodeProcessProviderParse error parsing process output
	ErrCodeProcessProviderParse = "ProcessProviderParseError"

	// ErrCodeProcessProviderVersion version error in output
	ErrCodeProcessProviderVersion = "ProcessProviderVersionError"

	// ErrCodeProcessProviderRequired required attribute missing in output
	ErrCodeProcessProviderRequired = "ProcessProviderRequiredError"

	// ErrCodeProcessProviderExecution execution of command failed
	ErrCodeProcessProviderExecution = "ProcessProviderExecutionError"

	// errMsgProcessProviderTimeout process took longer than allowed
	errMsgProcessProviderTimeout = "credential process timed out"

	// errMsgProcessProviderProcess process error
	errMsgProcessProviderProcess = "error in credential_process"

	// errMsgProcessProviderParse problem parsing output
	errMsgProcessProviderParse = "parse failed of credential_process output"

	// errMsgProcessProviderVersion version error in output
	errMsgProcessProviderVersion = "wrong version in process output (not 1)"

	// errMsgProcessProviderMissKey missing access key id in output
	errMsgProcessProviderMissKey = "missing AccessKeyId in process output"

	// errMsgProcessProviderMissSecret missing secret acess key in output
	errMsgProcessProviderMissSecret = "missing SecretAccessKey in process output"

	// errMsgProcessProviderPrepareCmd prepare of command failed
	errMsgProcessProviderPrepareCmd = "failed to prepare command"

	// errMsgProcessProviderEmptyCmd command must not be empty
	errMsgProcessProviderEmptyCmd = "command must not be empty"

	// errMsgProcessProviderPipe failed to initialize pipe
	errMsgProcessProviderPipe = "failed to initialize pipe"

	// DefaultDuration is the default amount of time in minutes that the
	// credentials will be valid for.
	DefaultDuration = time.Duration(15) * time.Minute

	// DefaultBufSize limits buffer size from growing to an enormous
	// amount due to a faulty process.
	DefaultBufSize = 1024

	// DefaultTimeout default limit on time a process can run.
	DefaultTimeout = time.Duration(1) * time.Minute
)

// ProcessProvider satisfies the credentials.Provider interface, and is a
// client to retrieve credentials from a process.
type ProcessProvider struct {
	staticCreds bool
	credentials.Expiry
	originalCommand []string

	// Expiry duration of the credentials. Defaults to 15 minutes if not set.
	Duration time.Duration

	// ExpiryWindow will allow the credentials to trigger refreshing prior to
	// the credentials actually expiring. This is beneficial so race conditions
	// with expiring credentials do not cause request to fail unexpectedly
	// due to ExpiredTokenException exceptions.
	//
	// So a ExpiryWindow of 10s would cause calls to IsExpired() to return true
	// 10 seconds before the credentials are actually expired.
	//
	// If ExpiryWindow is 0 or less it will be ignored.
	ExpiryWindow time.Duration

	// A string representing an os command that should return a JSON with
	// credential information.
	command *exec.Cmd

	// MaxBufSize limits memory usage from growing to an enormous
	// amount due to a faulty process.
	MaxBufSize int

	// Timeout limits the time a process can run.
	Timeout time.Duration
}

// NewCredentials returns a pointer to a new Credentials object wrapping the
// ProcessProvider. The credentials will expire every 15 minutes by default.
func NewCredentials(command string, options ...func(*ProcessProvider)) *credentials.Credentials {
	p := &ProcessProvider{
		command:    exec.Command(command),
		Duration:   DefaultDuration,
		Timeout:    DefaultTimeout,
		MaxBufSize: DefaultBufSize,
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

// NewCredentialsTimeout returns a pointer to a new Credentials object with
// the specified command and timeout, and default duration and max buffer size.
func NewCredentialsTimeout(command string, timeout time.Duration) *credentials.Credentials {
	p := NewCredentials(command, func(opt *ProcessProvider) {
		opt.Timeout = timeout
	})

	return p
}

// NewCredentialsCommand returns a pointer to a new Credentials object with
// the specified command, and default timeout, duration and max buffer size.
func NewCredentialsCommand(command *exec.Cmd, options ...func(*ProcessProvider)) *credentials.Credentials {
	p := &ProcessProvider{
		command:    command,
		Duration:   DefaultDuration,
		Timeout:    DefaultTimeout,
		MaxBufSize: DefaultBufSize,
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

type credentialProcessResponse struct {
	Version         int
	AccessKeyID     string `json:"AccessKeyId"`
	SecretAccessKey string
	SessionToken    string
	Expiration      *time.Time
}

// Retrieve executes the 'credential_process' and returns the credentials.
func (p *ProcessProvider) Retrieve() (credentials.Value, error) {
	out, err := p.executeCredentialProcess()
	if err != nil {
		return credentials.Value{ProviderName: ProviderName}, err
	}

	// Serialize and validate response
	resp := &credentialProcessResponse{}
	if err = json.Unmarshal(out, resp); err != nil {
		return credentials.Value{ProviderName: ProviderName}, awserr.New(
			ErrCodeProcessProviderParse,
			fmt.Sprintf("%s: %s", errMsgProcessProviderParse, string(out)),
			err)
	}

	if resp.Version != 1 {
		return credentials.Value{ProviderName: ProviderName}, awserr.New(
			ErrCodeProcessProviderVersion,
			errMsgProcessProviderVersion,
			nil)
	}

	if len(resp.AccessKeyID) == 0 {
		return credentials.Value{ProviderName: ProviderName}, awserr.New(
			ErrCodeProcessProviderRequired,
			errMsgProcessProviderMissKey,
			nil)
	}

	if len(resp.SecretAccessKey) == 0 {
		return credentials.Value{ProviderName: ProviderName}, awserr.New(
			ErrCodeProcessProviderRequired,
			errMsgProcessProviderMissSecret,
			nil)
	}

	// Handle expiration
	p.staticCreds = resp.Expiration == nil
	if resp.Expiration != nil {
		p.SetExpiration(*resp.Expiration, p.ExpiryWindow)
	}

	return credentials.Value{
		ProviderName:    ProviderName,
		AccessKeyID:     resp.AccessKeyID,
		SecretAccessKey: resp.SecretAccessKey,
		SessionToken:    resp.SessionToken,
	}, nil
}

// IsExpired returns true if the credentials retrieved are expired, or not yet
// retrieved.
func (p *ProcessProvider) IsExpired() bool {
	if p.staticCreds {
		return false
	}
	return p.Expiry.IsExpired()
}

// prepareCommand prepares the command to be executed.
func (p *ProcessProvider) prepareCommand() error {

	var cmdArgs []string
	if runtime.GOOS == "windows" {
		cmdArgs = []string{"cmd.exe", "/C"}
	} else {
		cmdArgs = []string{"sh", "-c"}
	}

	if len(p.originalCommand) == 0 {
		p.originalCommand = make([]string, len(p.command.Args))
		copy(p.originalCommand, p.command.Args)

		// check for empty command because it succeeds
		if len(strings.TrimSpace(p.originalCommand[0])) < 1 {
			return awserr.New(
				ErrCodeProcessProviderExecution,
				fmt.Sprintf(
					"%s: %s",
					errMsgProcessProviderPrepareCmd,
					errMsgProcessProviderEmptyCmd),
				nil)
		}
	}

	cmdArgs = append(cmdArgs, p.originalCommand...)
	p.command = exec.Command(cmdArgs[0], cmdArgs[1:]...)
	p.command.Env = os.Environ()

	return nil
}

// executeCredentialProcess starts the credential process on the OS and
// returns the results or an error.
func (p *ProcessProvider) executeCredentialProcess() ([]byte, error) {

	if err := p.prepareCommand(); err != nil {
		return nil, err
	}

	// Setup the pipes
	outReadPipe, outWritePipe, err := os.Pipe()
	if err != nil {
		return nil, awserr.New(
			ErrCodeProcessProviderExecution,
			errMsgProcessProviderPipe,
			err)
	}

	p.command.Stderr = os.Stderr    // display stderr on console for MFA
	p.command.Stdout = outWritePipe // get creds json on process's stdout
	p.command.Stdin = os.Stdin      // enable stdin for MFA

	output := bytes.NewBuffer(make([]byte, 0, p.MaxBufSize))

	stdoutCh := make(chan error, 1)
	go readInput(
		io.LimitReader(outReadPipe, int64(p.MaxBufSize)),
		output,
		stdoutCh)

	execCh := make(chan error, 1)
	go executeCommand(*p.command, execCh)

	finished := false
	var errors []error
	for !finished {
		select {
		case readError := <-stdoutCh:
			errors = appendError(errors, readError)
			finished = true
		case execError := <-execCh:
			err := outWritePipe.Close()
			errors = appendError(errors, err)
			errors = appendError(errors, execError)
			if errors != nil {
				return output.Bytes(), awserr.NewBatchError(
					ErrCodeProcessProviderExecution,
					errMsgProcessProviderProcess,
					errors)
			}
		case <-time.After(p.Timeout):
			finished = true
			return output.Bytes(), awserr.NewBatchError(
				ErrCodeProcessProviderExecution,
				errMsgProcessProviderTimeout,
				errors) // errors can be nil
		}
	}

	out := output.Bytes()

	if runtime.GOOS == "windows" {
		// windows adds slashes to quotes
		out = []byte(strings.Replace(string(out), `\"`, `"`, -1))
	}

	return out, nil
}

// appendError conveniently checks for nil before appending slice
func appendError(errors []error, err error) []error {
	if err != nil {
		return append(errors, err)
	}
	return errors
}

func executeCommand(cmd exec.Cmd, exec chan error) {
	// Start the command
	err := cmd.Start()
	if err == nil {
		err = cmd.Wait()
	}

	exec <- err
}

func readInput(r io.Reader, w io.Writer, read chan error) {
	tee := io.TeeReader(r, w)

	_, err := ioutil.ReadAll(tee)

	if err == io.EOF {
		err = nil
	}

	read <- err // will only arrive here when write end of pipe is closed
}
