package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/template"

	"github.com/docker/docker/api/types"
	flag "github.com/docker/docker/pkg/mflag"
)

// CmdInspect displays low-level information on one or more containers or images.
//
// Usage: docker inspect [OPTIONS] CONTAINER|IMAGE [CONTAINER|IMAGE...]
func (cli *DockerCli) CmdInspect(args ...string) error {
	cmd := cli.Subcmd("inspect", []string{"CONTAINER|IMAGE [CONTAINER|IMAGE...]"}, "Return low-level information on a container or image", true)
	tmplStr := cmd.String([]string{"f", "#format", "-format"}, "", "Format the output using the given go template")
	inspectType := cmd.String([]string{"-type"}, "", "Return JSON for specified type, (e.g image or container)")
	cmd.Require(flag.Min, 1)

	cmd.ParseFlags(args, true)

	var tmpl *template.Template
	var err error
	var obj []byte

	if *tmplStr != "" {
		if tmpl, err = template.New("").Funcs(funcMap).Parse(*tmplStr); err != nil {
			return StatusError{StatusCode: 64,
				Status: "Template parsing error: " + err.Error()}
		}
	}

	if *inspectType != "" && *inspectType != "container" && *inspectType != "image" {
		return fmt.Errorf("%q is not a valid value for --type", *inspectType)
	}

	indented := new(bytes.Buffer)
	indented.WriteString("[\n")
	status := 0
	isImage := false

	for _, name := range cmd.Args() {

		if *inspectType == "" || *inspectType == "container" {
			obj, _, err = readBody(cli.call("GET", "/containers/"+name+"/json", nil, nil))
			if err != nil && *inspectType == "container" {
				if strings.Contains(err.Error(), "No such") {
					fmt.Fprintf(cli.err, "Error: No such container: %s\n", name)
				} else {
					fmt.Fprintf(cli.err, "%s", err)
				}
				status = 1
				continue
			}
		}

		if obj == nil && (*inspectType == "" || *inspectType == "image") {
			obj, _, err = readBody(cli.call("GET", "/images/"+name+"/json", nil, nil))
			isImage = true
			if err != nil {
				if strings.Contains(err.Error(), "No such") {
					if *inspectType == "" {
						fmt.Fprintf(cli.err, "Error: No such image or container: %s\n", name)
					} else {
						fmt.Fprintf(cli.err, "Error: No such image: %s\n", name)
					}
				} else {
					fmt.Fprintf(cli.err, "%s", err)
				}
				status = 1
				continue
			}

		}

		if tmpl == nil {
			if err := json.Indent(indented, obj, "", "    "); err != nil {
				fmt.Fprintf(cli.err, "%s\n", err)
				status = 1
				continue
			}
		} else {
			rdr := bytes.NewReader(obj)
			dec := json.NewDecoder(rdr)

			if isImage {
				inspPtr := types.ImageInspect{}
				if err := dec.Decode(&inspPtr); err != nil {
					fmt.Fprintf(cli.err, "%s\n", err)
					status = 1
					continue
				}
				if err := tmpl.Execute(cli.out, inspPtr); err != nil {
					rdr.Seek(0, 0)
					var raw interface{}
					if err := dec.Decode(&raw); err != nil {
						return err
					}
					if err = tmpl.Execute(cli.out, raw); err != nil {
						return err
					}
				}
			} else {
				inspPtr := types.ContainerJSON{}
				if err := dec.Decode(&inspPtr); err != nil {
					fmt.Fprintf(cli.err, "%s\n", err)
					status = 1
					continue
				}
				if err := tmpl.Execute(cli.out, inspPtr); err != nil {
					rdr.Seek(0, 0)
					var raw interface{}
					if err := dec.Decode(&raw); err != nil {
						return err
					}
					if err = tmpl.Execute(cli.out, raw); err != nil {
						return err
					}
				}
			}
			cli.out.Write([]byte{'\n'})
		}
		indented.WriteString(",")
	}

	if indented.Len() > 1 {
		// Remove trailing ','
		indented.Truncate(indented.Len() - 1)
	}
	indented.WriteString("]\n")

	if tmpl == nil {
		// Note that we will always write "[]" when "-f" isn't specified,
		// to make sure the output would always be array, see
		// https://github.com/docker/docker/pull/9500#issuecomment-65846734
		if _, err := io.Copy(cli.out, indented); err != nil {
			return err
		}
	}

	if status != 0 {
		return StatusError{StatusCode: status}
	}
	return nil
}
