package client

import (
	"io"
	"os"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdLoad loads an image from a tar archive.
//
// The tar archive is read from STDIN by default, or from a tar archive file.
//
// Usage: docker load [OPTIONS]
func (cli *DockerCli) CmdLoad(args ...string) error {
	cmd := cli.Subcmd("load", nil, "Load an image from a tar archive or STDIN", true)
	infile := cmd.String([]string{"i", "-input"}, "", "Read from a tar archive file, instead of STDIN")
	cmd.Require(flag.Exact, 0)

	cmd.ParseFlags(args, true)

	var (
		input io.Reader = cli.in
		err   error
	)
	if *infile != "" {
		input, err = os.Open(*infile)
		if err != nil {
			return err
		}
	}
	sopts := &streamOpts{
		rawTerminal: true,
		in:          input,
		out:         cli.out,
	}
	if _, err := cli.stream("POST", "/images/load", sopts); err != nil {
		return err
	}
	return nil
}
