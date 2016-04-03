package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/docker/docker"
	"os"
	"strings"
	"text/template"
)

var templates = map[string]string{

	"upstart": `description "{{.description}}"
author "{{.author}}"
start on filesystem and started lxc-net and started docker
stop on runlevel [!2345]
respawn
exec /home/vagrant/goroot/bin/docker start -a {{.container_id}}
`,

	"systemd": `[Unit]
	Description={{.description}}
	Author={{.author}}
	After=docker.service

[Service]
	Restart=always
	ExecStart=/usr/bin/docker start -a {{.container_id}}
	ExecStop=/usr/bin/docker stop -t 2 {{.container_id}}

[Install]
	WantedBy=local.target
`,
}

func main() {
	// Parse command line for custom options
	kind := flag.String("t", "upstart", "Type of manager requested")
	author := flag.String("a", "<none>", "Author of the image")
	description := flag.String("d", "<none>", "Description of the image")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "\nUsage: manager <container id>\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// We require at least the container ID
	if flag.NArg() != 1 {
		println(flag.NArg())
		flag.Usage()
		return
	}

	// Check that the requested process manager is supported
	if _, exists := templates[*kind]; !exists {
		panic("Unknown script template")
	}

	// Load the requested template
	tpl, err := template.New("processManager").Parse(templates[*kind])
	if err != nil {
		panic(err)
	}

	// Create stdout/stderr buffers
	bufOut := bytes.NewBuffer(nil)
	bufErr := bytes.NewBuffer(nil)

	// Instanciate the Docker CLI
	cli := docker.NewDockerCli(nil, bufOut, bufErr, "unix", "/var/run/docker.sock", false, nil)
	// Retrieve the container info
	if err := cli.CmdInspect(flag.Arg(0)); err != nil {
		// As of docker v0.6.3, CmdInspect always returns nil
		panic(err)
	}

	// If there is nothing in the error buffer, then the Docker daemon is there and the container has been found
	if bufErr.Len() == 0 {
		// Unmarshall the resulting container data
		c := []*docker.Container{{}}
		if err := json.Unmarshal(bufOut.Bytes(), &c); err != nil {
			panic(err)
		}
		// Reset the buffers
		bufOut.Reset()
		bufErr.Reset()
		// Retrieve the info of the linked image
		if err := cli.CmdInspect(c[0].Image); err != nil {
			panic(err)
		}
		// If there is nothing in the error buffer, then the image has been found.
		if bufErr.Len() == 0 {
			// Unmarshall the resulting image data
			img := []*docker.Image{{}}
			if err := json.Unmarshal(bufOut.Bytes(), &img); err != nil {
				panic(err)
			}
			// If no author has been set, use the one from the image
			if *author == "<none>" && img[0].Author != "" {
				*author = strings.Replace(img[0].Author, "\"", "", -1)
			}
			// If no description has been set, use the comment from the image
			if *description == "<none>" && img[0].Comment != "" {
				*description = strings.Replace(img[0].Comment, "\"", "", -1)
			}
		}
	}

	/// Old version: Wrtie the resulting script to file
	// f, err := os.OpenFile(kind, os.O_CREATE|os.O_WRONLY, 0755)
	// if err != nil {
	// 	panic(err)
	// }
	// defer f.Close()

	// Create a map with needed data
	data := map[string]string{
		"author":       *author,
		"description":  *description,
		"container_id": flag.Arg(0),
	}

	// Process the template and output it on Stdout
	if err := tpl.Execute(os.Stdout, data); err != nil {
		panic(err)
	}
}
