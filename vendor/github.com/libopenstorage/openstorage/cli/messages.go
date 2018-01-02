package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/codegangsta/cli"
	"github.com/golang/protobuf/proto"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/jsonpb"
)

var (
	marshaler = &jsonpb.Marshaler{EnumsAsSimpleStrings: true, Indent: " "}
)

// Format standardizes the screen output of commands.
type Format struct {
	Cmd     string       `json:"cmd,omitempty"`
	Status  string       `json:"status,omitempty"`
	Err     string       `json:"error,omitempty"`
	Desc    string       `json:"desc,omitempty"`
	UUID    []string     `json:"uuid,omitempty"`
	Result  interface{}  `json:"result,omitempty"`
	Cluster *api.Cluster `json:"cluster,omitempty"`
}

func exitCli() {
	os.Exit(1)
}

func missingParameter(c *cli.Context, cmd string, param string, desc string) {
	fmtOutput(c,
		&Format{
			Cmd:  cmd,
			Err:  fmt.Sprintf("missing parameter %q", param),
			Desc: desc,
		})
	exitCli()
}

func incorrectUsage(c *cli.Context, cmd string, desc string) {
	fmtOutput(c,
		&Format{
			Cmd:  cmd,
			Err:  fmt.Sprintf("incorrect usage"),
			Desc: desc,
		})
	exitCli()
}

func badParameter(c *cli.Context, cmd string, param string, desc string) {
	fmtOutput(c,
		&Format{
			Cmd:  cmd,
			Err:  fmt.Sprintf("missing parameter %q", param),
			Desc: desc,
		})
	exitCli()
}

func cmdError(c *cli.Context, cmd string, err error) {
	fmtOutput(c,
		&Format{
			Cmd: cmd,
			Err: err.Error(),
		})
	exitCli()
}

func cmdErrorBody(c *cli.Context, cmd string, err error, body string) {
	body = strings.Replace(body, "\n", " ", -1)
	body = strings.Replace(body, "\"", "", -1)
	fmtOutput(c,
		&Format{
			Cmd:  cmd,
			Err:  err.Error(),
			Desc: body,
		})
	exitCli()
}

func cmdOutput(c *cli.Context, body interface{}) {
	fmt.Printf("%+v\n", cmdMarshal(body))
}

func cmdMarshal(body interface{}) string {
	b, _ := json.MarshalIndent(body, "", " ")
	return string(b)
}

func cmdMarshalProto(message proto.Message, raw bool) string {
	if raw {
		return cmdMarshal(message)
	}
	s, _ := marshaler.MarshalToString(message)
	return s
}

func cmdOutputProto(message proto.Message, j bool) {
	fmt.Println(cmdMarshalProto(message, j))
}

func fmtOutput(c *cli.Context, format *Format) {
	jsonOut := c.GlobalBool("json")
	outFd := os.Stdout

	if format.Err != "" {
		outFd = os.Stderr
	}

	if jsonOut {
		b, _ := json.MarshalIndent(format, "", " ")
		fmt.Fprintf(outFd, "%+v\n", string(b))
		return
	}

	if format.Err == "" {
		if format.Result == nil {
			for _, v := range format.UUID {
				fmt.Fprintln(outFd, v)
			}
			return
		}
		b, _ := json.MarshalIndent(format.Result, "", " ")
		fmt.Fprintf(outFd, "%+v\n", string(b))
		return
	}

	if format.Desc != "" {
		fmt.Fprintf(outFd, "%s: %v - %s\n", format.Cmd, format.Err,
			format.Desc)
		return
	}

	fmt.Fprintf(outFd, "%s: %v\n", format.Cmd, format.Err)
}
