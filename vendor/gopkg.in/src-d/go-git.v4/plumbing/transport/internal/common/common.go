// Package common implements the git pack protocol with a pluggable transport.
// This is a low-level package to implement new transports. Use a concrete
// implementation instead (e.g. http, file, ssh).
//
// A simple example of usage can be found in the file package.
package common

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	stdioutil "io/ioutil"
	"strings"
	"time"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

const (
	readErrorSecondsTimeout = 10
)

var (
	ErrTimeoutExceeded = errors.New("timeout exceeded")
)

// Commander creates Command instances. This is the main entry point for
// transport implementations.
type Commander interface {
	// Command creates a new Command for the given git command and
	// endpoint. cmd can be git-upload-pack or git-receive-pack. An
	// error should be returned if the endpoint is not supported or the
	// command cannot be created (e.g. binary does not exist, connection
	// cannot be established).
	Command(cmd string, ep transport.Endpoint, auth transport.AuthMethod) (Command, error)
}

// Command is used for a single command execution.
// This interface is modeled after exec.Cmd and ssh.Session in the standard
// library.
type Command interface {
	// StderrPipe returns a pipe that will be connected to the command's
	// standard error when the command starts. It should not be called after
	// Start.
	StderrPipe() (io.Reader, error)
	// StdinPipe returns a pipe that will be connected to the command's
	// standard input when the command starts. It should not be called after
	// Start. The pipe should be closed when no more input is expected.
	StdinPipe() (io.WriteCloser, error)
	// StdoutPipe returns a pipe that will be connected to the command's
	// standard output when the command starts. It should not be called after
	// Start.
	StdoutPipe() (io.Reader, error)
	// Start starts the specified command. It does not wait for it to
	// complete.
	Start() error
	// Close closes the command and releases any resources used by it. It
	// will block until the command exits.
	Close() error
}

// CommandKiller expands the Command interface, enableing it for being killed.
type CommandKiller interface {
	// Kill and close the session whatever the state it is. It will block until
	// the command is terminated.
	Kill() error
}

type client struct {
	cmdr Commander
}

// NewClient creates a new client using the given Commander.
func NewClient(runner Commander) transport.Transport {
	return &client{runner}
}

// NewUploadPackSession creates a new UploadPackSession.
func (c *client) NewUploadPackSession(ep transport.Endpoint, auth transport.AuthMethod) (
	transport.UploadPackSession, error) {

	return c.newSession(transport.UploadPackServiceName, ep, auth)
}

// NewReceivePackSession creates a new ReceivePackSession.
func (c *client) NewReceivePackSession(ep transport.Endpoint, auth transport.AuthMethod) (
	transport.ReceivePackSession, error) {

	return c.newSession(transport.ReceivePackServiceName, ep, auth)
}

type session struct {
	Stdin   io.WriteCloser
	Stdout  io.Reader
	Command Command

	isReceivePack bool
	advRefs       *packp.AdvRefs
	packRun       bool
	finished      bool
	firstErrLine  chan string
}

func (c *client) newSession(s string, ep transport.Endpoint, auth transport.AuthMethod) (*session, error) {
	cmd, err := c.cmdr.Command(s, ep, auth)
	if err != nil {
		return nil, err
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	return &session{
		Stdin:         stdin,
		Stdout:        stdout,
		Command:       cmd,
		firstErrLine:  c.listenFirstError(stderr),
		isReceivePack: s == transport.ReceivePackServiceName,
	}, nil
}

func (c *client) listenFirstError(r io.Reader) chan string {
	if r == nil {
		return nil
	}

	errLine := make(chan string, 1)
	go func() {
		s := bufio.NewScanner(r)
		if s.Scan() {
			errLine <- s.Text()
		} else {
			close(errLine)
		}

		_, _ = io.Copy(stdioutil.Discard, r)
	}()

	return errLine
}

// AdvertisedReferences retrieves the advertised references from the server.
func (s *session) AdvertisedReferences() (*packp.AdvRefs, error) {
	if s.advRefs != nil {
		return s.advRefs, nil
	}

	ar := packp.NewAdvRefs()
	if err := ar.Decode(s.Stdout); err != nil {
		if err := s.handleAdvRefDecodeError(err); err != nil {
			return nil, err
		}
	}

	transport.FilterUnsupportedCapabilities(ar.Capabilities)
	s.advRefs = ar
	return ar, nil
}

func (s *session) handleAdvRefDecodeError(err error) error {
	// If repository is not found, we get empty stdout and server writes an
	// error to stderr.
	if err == packp.ErrEmptyInput {
		s.finished = true
		if err := s.checkNotFoundError(); err != nil {
			return err
		}

		return io.ErrUnexpectedEOF
	}

	// For empty (but existing) repositories, we get empty advertised-references
	// message. But valid. That is, it includes at least a flush.
	if err == packp.ErrEmptyAdvRefs {
		// Empty repositories are valid for git-receive-pack.
		if s.isReceivePack {
			return nil
		}

		if err := s.finish(); err != nil {
			return err
		}

		return transport.ErrEmptyRemoteRepository
	}

	// Some server sends the errors as normal content (git protocol), so when
	// we try to decode it fails, we need to check the content of it, to detect
	// not found errors
	if uerr, ok := err.(*packp.ErrUnexpectedData); ok {
		if isRepoNotFoundError(string(uerr.Data)) {
			return transport.ErrRepositoryNotFound
		}
	}

	return err
}

// UploadPack performs a request to the server to fetch a packfile. A reader is
// returned with the packfile content. The reader must be closed after reading.
func (s *session) UploadPack(ctx context.Context, req *packp.UploadPackRequest) (*packp.UploadPackResponse, error) {
	if req.IsEmpty() {
		return nil, transport.ErrEmptyUploadPackRequest
	}

	if err := req.Validate(); err != nil {
		return nil, err
	}

	if _, err := s.AdvertisedReferences(); err != nil {
		return nil, err
	}

	s.packRun = true

	in := s.StdinContext(ctx)
	out := s.StdoutContext(ctx)

	if err := uploadPack(in, out, req); err != nil {
		return nil, err
	}

	r, err := ioutil.NonEmptyReader(out)
	if err == ioutil.ErrEmptyReader {
		if c, ok := s.Stdout.(io.Closer); ok {
			_ = c.Close()
		}

		return nil, transport.ErrEmptyUploadPackRequest
	}

	if err != nil {
		return nil, err
	}

	rc := ioutil.NewReadCloser(r, s)
	return DecodeUploadPackResponse(rc, req)
}

func (s *session) StdinContext(ctx context.Context) io.WriteCloser {
	return ioutil.NewWriteCloserOnError(
		ioutil.NewContextWriteCloser(ctx, s.Stdin),
		s.onError,
	)
}

func (s *session) StdoutContext(ctx context.Context) io.Reader {
	return ioutil.NewReaderOnError(
		ioutil.NewContextReader(ctx, s.Stdout),
		s.onError,
	)
}

func (s *session) onError(err error) {
	if k, ok := s.Command.(CommandKiller); ok {
		_ = k.Kill()
	}

	_ = s.Close()
}

func (s *session) ReceivePack(ctx context.Context, req *packp.ReferenceUpdateRequest) (*packp.ReportStatus, error) {
	if _, err := s.AdvertisedReferences(); err != nil {
		return nil, err
	}

	s.packRun = true

	w := s.StdinContext(ctx)
	if err := req.Encode(w); err != nil {
		return nil, err
	}

	if err := w.Close(); err != nil {
		return nil, err
	}

	if !req.Capabilities.Supports(capability.ReportStatus) {
		// If we don't have report-status, we can only
		// check return value error.
		return nil, s.Command.Close()
	}

	r := s.StdoutContext(ctx)

	var d *sideband.Demuxer
	if req.Capabilities.Supports(capability.Sideband64k) {
		d = sideband.NewDemuxer(sideband.Sideband64k, r)
	} else if req.Capabilities.Supports(capability.Sideband) {
		d = sideband.NewDemuxer(sideband.Sideband, r)
	}
	if d != nil {
		d.Progress = req.Progress
		r = d
	}

	report := packp.NewReportStatus()
	if err := report.Decode(r); err != nil {
		return nil, err
	}

	if err := report.Error(); err != nil {
		defer s.Close()
		return report, err
	}

	return report, s.Command.Close()
}

func (s *session) finish() error {
	if s.finished {
		return nil
	}

	s.finished = true

	// If we did not run a upload/receive-pack, we close the connection
	// gracefully by sending a flush packet to the server. If the server
	// operates correctly, it will exit with status 0.
	if !s.packRun {
		_, err := s.Stdin.Write(pktline.FlushPkt)
		return err
	}

	return nil
}

func (s *session) Close() (err error) {
	err = s.finish()

	defer ioutil.CheckClose(s.Command, &err)
	return
}

func (s *session) checkNotFoundError() error {
	t := time.NewTicker(time.Second * readErrorSecondsTimeout)
	defer t.Stop()

	select {
	case <-t.C:
		return ErrTimeoutExceeded
	case line, ok := <-s.firstErrLine:
		if !ok {
			return nil
		}

		if isRepoNotFoundError(line) {
			return transport.ErrRepositoryNotFound
		}

		return fmt.Errorf("unknown error: %s", line)
	}
}

var (
	githubRepoNotFoundErr      = "ERROR: Repository not found."
	bitbucketRepoNotFoundErr   = "conq: repository does not exist."
	localRepoNotFoundErr       = "does not appear to be a git repository"
	gitProtocolNotFoundErr     = "ERR \n  Repository not found."
	gitProtocolNoSuchErr       = "ERR no such repository"
	gitProtocolAccessDeniedErr = "ERR access denied"
)

func isRepoNotFoundError(s string) bool {
	if strings.HasPrefix(s, githubRepoNotFoundErr) {
		return true
	}

	if strings.HasPrefix(s, bitbucketRepoNotFoundErr) {
		return true
	}

	if strings.HasSuffix(s, localRepoNotFoundErr) {
		return true
	}

	if strings.HasPrefix(s, gitProtocolNotFoundErr) {
		return true
	}

	if strings.HasPrefix(s, gitProtocolNoSuchErr) {
		return true
	}

	if strings.HasPrefix(s, gitProtocolAccessDeniedErr) {
		return true
	}

	return false
}

var (
	nak = []byte("NAK")
	eol = []byte("\n")
)

// uploadPack implements the git-upload-pack protocol.
func uploadPack(w io.WriteCloser, r io.Reader, req *packp.UploadPackRequest) error {
	// TODO support multi_ack mode
	// TODO support multi_ack_detailed mode
	// TODO support acks for common objects
	// TODO build a proper state machine for all these processing options

	if err := req.UploadRequest.Encode(w); err != nil {
		return fmt.Errorf("sending upload-req message: %s", err)
	}

	if err := req.UploadHaves.Encode(w, true); err != nil {
		return fmt.Errorf("sending haves message: %s", err)
	}

	if err := sendDone(w); err != nil {
		return fmt.Errorf("sending done message: %s", err)
	}

	if err := w.Close(); err != nil {
		return fmt.Errorf("closing input: %s", err)
	}

	return nil
}

func sendDone(w io.Writer) error {
	e := pktline.NewEncoder(w)

	return e.Encodef("done\n")
}

// DecodeUploadPackResponse decodes r into a new packp.UploadPackResponse
func DecodeUploadPackResponse(r io.ReadCloser, req *packp.UploadPackRequest) (
	*packp.UploadPackResponse, error,
) {
	res := packp.NewUploadPackResponse(req)
	if err := res.Decode(r); err != nil {
		return nil, fmt.Errorf("error decoding upload-pack response: %s", err)
	}

	return res, nil
}
