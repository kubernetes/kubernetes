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

package guest

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
)

type run struct {
	*GuestFlag

	data    string
	verbose bool
	dir     string
	vars    env
}

func init() {
	cli.Register("guest.run", &run{})
}

func (cmd *run) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.GuestFlag, ctx = newGuestFlag(ctx)
	cmd.GuestFlag.Register(ctx, f)

	f.StringVar(&cmd.data, "d", "", "Input data")
	f.BoolVar(&cmd.verbose, "v", false, "Verbose")
	f.StringVar(&cmd.dir, "C", "", "The absolute path of the working directory for the program to start")
	f.Var(&cmd.vars, "e", "Set environment variable or HTTP header")
}

func (cmd *run) Process(ctx context.Context) error {
	if err := cmd.GuestFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *run) Usage() string {
	return "NAME [ARG]..."
}

func (cmd *run) Description() string {
	return `Run program NAME in VM and display output.

This command depends on govmomi/toolbox running in the VM guest and does not work with standard VMware tools.

If the program NAME is an HTTP verb, the toolbox's http.RoundTripper will be used as the HTTP transport.

Examples:
  govc guest.run -vm $name kubectl get pods
  govc guest.run -vm $name -d - kubectl create -f - <svc.json
  govc guest.run -vm $name kubectl delete pod,service my-service
  govc guest.run -vm $name GET http://localhost:8080/api/v1/nodes
  govc guest.run -vm $name -e Content-Type:application/json -d - POST http://localhost:8080/api/v1/namespaces/default/pods <svc.json
  govc guest.run -vm $name DELETE http://localhost:8080/api/v1/namespaces/default/services/my-service`
}

func (cmd *run) do(c *http.Client, req *http.Request) error {
	for _, v := range cmd.vars {
		h := strings.SplitN(v, ":", 2)
		if len(h) != 2 {
			return fmt.Errorf("invalid header: %q", v)
		}

		req.Header.Set(strings.TrimSpace(h[0]), strings.TrimSpace(h[1]))
	}

	res, err := c.Do(req)
	if err != nil {
		return err
	}

	if cmd.verbose {
		return res.Write(cmd.Out)
	}

	_, err = io.Copy(cmd.Out, res.Body)

	_ = res.Body.Close()

	return err
}

func (cmd *run) Run(ctx context.Context, f *flag.FlagSet) error {
	name := f.Arg(0)

	tc, err := cmd.Toolbox()
	if err != nil {
		return err
	}

	hc := &http.Client{
		Transport: tc,
	}

	switch name {
	case "HEAD", "GET", "DELETE":
		req, err := http.NewRequest(name, f.Arg(1), nil)
		if err != nil {
			return err
		}

		return cmd.do(hc, req)
	case "POST", "PUT":
		req, err := http.NewRequest(name, f.Arg(1), os.Stdin)
		if err != nil {
			return err
		}

		return cmd.do(hc, req)
	default:
		ecmd := &exec.Cmd{
			Path:   name,
			Args:   f.Args()[1:],
			Env:    cmd.vars,
			Dir:    cmd.dir,
			Stdout: os.Stdout,
			Stderr: os.Stderr,
		}

		switch cmd.data {
		case "":
		case "-":
			ecmd.Stdin = os.Stdin
		default:
			ecmd.Stdin = bytes.NewBuffer([]byte(cmd.data))
		}

		return tc.Run(ctx, ecmd)
	}
}
