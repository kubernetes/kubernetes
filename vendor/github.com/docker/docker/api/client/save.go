package client

import (
	"errors"
	"io"
	"net/url"
	"os"

	flag "github.com/docker/docker/pkg/mflag"
)

// CmdSave saves one or more images to a tar archive.
//
// The tar archive is written to STDOUT by default, or written to a file.
//
// Usage: docker save [OPTIONS] IMAGE [IMAGE...]
func (cli *DockerCli) CmdSave(args ...string) error {
	cmd := cli.Subcmd("save", []string{"IMAGE [IMAGE...]"}, "Save an image(s) to a tar archive (streamed to STDOUT by default)", true)
	outfile := cmd.String([]string{"o", "-output"}, "", "Write to an file, instead of STDOUT")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var (
		output io.Writer = cli.out
		err    error
	)
	if *outfile != "" {
		output, err = os.Create(*outfile)
		if err != nil {
			return err
		}
	} else if cli.isTerminalOut {
		return errors.New("Cowardly refusing to save to a terminal. Use the -o flag or redirect.")
	}

	sopts := &streamOpts{
		rawTerminal: true,
		out:         output,
	}

	if len(cmd.Args()) == 1 {
		image := cmd.Arg(0)
		if _, err := cli.stream("GET", "/images/"+image+"/get", sopts); err != nil {
			return err
		}
	} else {
		v := url.Values{}
		for _, arg := range cmd.Args() {
			v.Add("names", arg)
		}
		if _, err := cli.stream("GET", "/images/get?"+v.Encode(), sopts); err != nil {
			return err
		}
	}
	return nil
}
