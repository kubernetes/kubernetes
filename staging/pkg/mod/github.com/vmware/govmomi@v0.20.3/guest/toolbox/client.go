/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package toolbox

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"

	"github.com/vmware/govmomi/guest"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// Client attempts to expose guest.OperationsManager as idiomatic Go interfaces
type Client struct {
	ProcessManager *guest.ProcessManager
	FileManager    *guest.FileManager
	Authentication types.BaseGuestAuthentication
}

// procReader retries InitiateFileTransferFromGuest calls if toolbox is still running the process.
// See also: ProcessManager.Stat
func (c *Client) procReader(ctx context.Context, src string) (*types.FileTransferInformation, error) {
	for {
		info, err := c.FileManager.InitiateFileTransferFromGuest(ctx, c.Authentication, src)
		if err != nil {
			if soap.IsSoapFault(err) {
				if _, ok := soap.ToSoapFault(err).VimFault().(types.CannotAccessFile); ok {
					// We're not waiting in between retries since ProcessManager.Stat
					// has already waited.  In the case that this client was pointed at
					// standard vmware-tools, the types.NotFound fault would have been
					// returned since the file "/proc/$pid/stdout" does not exist - in
					// which case, we won't retry at all.
					continue
				}
			}

			return nil, err
		}

		return info, err
	}
}

// RoundTrip implements http.RoundTripper over vmx guest RPC.
// This transport depends on govmomi/toolbox running in the VM guest and does not work with standard VMware tools.
// Using this transport makes it is possible to connect to HTTP endpoints that are bound to the VM's loopback address.
// Note that the toolbox's http.RoundTripper only supports the "http" scheme, "https" is not supported.
func (c *Client) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "http" {
		return nil, fmt.Errorf("%q scheme not supported", req.URL.Scheme)
	}

	ctx := req.Context()

	req.Header.Set("Connection", "close") // we need the server to close the connection after 1 request

	spec := types.GuestProgramSpec{
		ProgramPath: "http.RoundTrip",
		Arguments:   req.URL.Host,
	}

	pid, err := c.ProcessManager.StartProgram(ctx, c.Authentication, &spec)
	if err != nil {
		return nil, err
	}

	dst := fmt.Sprintf("/proc/%d/stdin", pid)
	src := fmt.Sprintf("/proc/%d/stdout", pid)

	var buf bytes.Buffer
	err = req.Write(&buf)
	if err != nil {
		return nil, err
	}

	attr := new(types.GuestPosixFileAttributes)
	size := int64(buf.Len())

	url, err := c.FileManager.InitiateFileTransferToGuest(ctx, c.Authentication, dst, attr, size, true)
	if err != nil {
		return nil, err
	}

	vc := c.ProcessManager.Client()

	u, err := c.FileManager.TransferURL(ctx, url)
	if err != nil {
		return nil, err
	}

	p := soap.DefaultUpload
	p.ContentLength = size

	err = vc.Client.Upload(ctx, &buf, u, &p)
	if err != nil {
		return nil, err
	}

	info, err := c.procReader(ctx, src)
	if err != nil {
		return nil, err
	}

	u, err = c.FileManager.TransferURL(ctx, info.Url)
	if err != nil {
		return nil, err
	}

	f, _, err := vc.Client.Download(ctx, u, &soap.DefaultDownload)
	if err != nil {
		return nil, err
	}

	return http.ReadResponse(bufio.NewReader(f), req)
}

// Run implements exec.Cmd.Run over vmx guest RPC.
func (c *Client) Run(ctx context.Context, cmd *exec.Cmd) error {
	vc := c.ProcessManager.Client()

	spec := types.GuestProgramSpec{
		ProgramPath:      cmd.Path,
		Arguments:        strings.Join(cmd.Args, " "),
		EnvVariables:     cmd.Env,
		WorkingDirectory: cmd.Dir,
	}

	pid, serr := c.ProcessManager.StartProgram(ctx, c.Authentication, &spec)
	if serr != nil {
		return serr
	}

	if cmd.Stdin != nil {
		dst := fmt.Sprintf("/proc/%d/stdin", pid)

		var buf bytes.Buffer
		size, err := io.Copy(&buf, cmd.Stdin)
		if err != nil {
			return err
		}

		attr := new(types.GuestPosixFileAttributes)

		url, err := c.FileManager.InitiateFileTransferToGuest(ctx, c.Authentication, dst, attr, size, true)
		if err != nil {
			return err
		}

		u, err := c.FileManager.TransferURL(ctx, url)
		if err != nil {
			return err
		}

		p := soap.DefaultUpload
		p.ContentLength = size

		err = vc.Client.Upload(ctx, &buf, u, &p)
		if err != nil {
			return err
		}
	}

	names := []string{"out", "err"}

	for i, w := range []io.Writer{cmd.Stdout, cmd.Stderr} {
		if w == nil {
			continue
		}

		src := fmt.Sprintf("/proc/%d/std%s", pid, names[i])

		info, err := c.procReader(ctx, src)
		if err != nil {
			return err
		}

		u, err := c.FileManager.TransferURL(ctx, info.Url)
		if err != nil {
			return err
		}

		f, _, err := vc.Client.Download(ctx, u, &soap.DefaultDownload)
		if err != nil {
			return err
		}

		_, err = io.Copy(w, f)
		_ = f.Close()
		if err != nil {
			return err
		}
	}

	procs, err := c.ProcessManager.ListProcesses(ctx, c.Authentication, []int64{pid})
	if err != nil {
		return err
	}

	if len(procs) == 1 {
		rc := procs[0].ExitCode
		if rc != 0 {
			return fmt.Errorf("%s: exit %d", cmd.Path, rc)
		}
	}

	return nil
}

// archiveReader wraps an io.ReadCloser to support streaming download
// of a guest directory, stops reading once it sees the stream trailer.
// This is only useful when guest tools is the Go toolbox.
// The trailer is required since TransferFromGuest requires a Content-Length,
// which toolbox doesn't know ahead of time as the gzip'd tarball never touches the disk.
// We opted to wrap this here for now rather than guest.FileManager so
// DownloadFile can be also be used as-is to handle this use case.
type archiveReader struct {
	io.ReadCloser
}

var (
	gzipHeader    = []byte{0x1f, 0x8b, 0x08} // rfc1952 {ID1, ID2, CM}
	gzipHeaderLen = len(gzipHeader)
)

func (r *archiveReader) Read(buf []byte) (int, error) {
	nr, err := r.ReadCloser.Read(buf)

	// Stop reading if the last N bytes are the gzipTrailer
	if nr >= gzipHeaderLen {
		if bytes.Equal(buf[nr-gzipHeaderLen:nr], gzipHeader) {
			nr -= gzipHeaderLen
			err = io.EOF
		}
	}

	return nr, err
}

func isDir(src string) bool {
	u, err := url.Parse(src)
	if err != nil {
		return false
	}

	return strings.HasSuffix(u.Path, "/")
}

// Download initiates a file transfer from the guest
func (c *Client) Download(ctx context.Context, src string) (io.ReadCloser, int64, error) {
	vc := c.ProcessManager.Client()

	info, err := c.FileManager.InitiateFileTransferFromGuest(ctx, c.Authentication, src)
	if err != nil {
		return nil, 0, err
	}

	u, err := c.FileManager.TransferURL(ctx, info.Url)
	if err != nil {
		return nil, 0, err
	}

	p := soap.DefaultDownload

	f, n, err := vc.Download(ctx, u, &p)
	if err != nil {
		return nil, n, err
	}

	if strings.HasPrefix(src, "/archive:/") || isDir(src) {
		f = &archiveReader{ReadCloser: f} // look for the gzip trailer
	}

	return f, n, nil
}

// Upload transfers a file to the guest
func (c *Client) Upload(ctx context.Context, src io.Reader, dst string, p soap.Upload, attr types.BaseGuestFileAttributes, force bool) error {
	vc := c.ProcessManager.Client()

	var err error

	if p.ContentLength == 0 { // Content-Length is required
		switch r := src.(type) {
		case *bytes.Buffer:
			p.ContentLength = int64(r.Len())
		case *bytes.Reader:
			p.ContentLength = int64(r.Len())
		case *strings.Reader:
			p.ContentLength = int64(r.Len())
		case *os.File:
			info, serr := r.Stat()
			if serr != nil {
				return serr
			}

			p.ContentLength = info.Size()
		}

		if p.ContentLength == 0 { // os.File for example could be a device (stdin)
			buf := new(bytes.Buffer)

			p.ContentLength, err = io.Copy(buf, src)
			if err != nil {
				return err
			}

			src = buf
		}
	}

	url, err := c.FileManager.InitiateFileTransferToGuest(ctx, c.Authentication, dst, attr, p.ContentLength, force)
	if err != nil {
		return err
	}

	u, err := c.FileManager.TransferURL(ctx, url)
	if err != nil {
		return err
	}

	return vc.Client.Upload(ctx, src, u, &p)
}
