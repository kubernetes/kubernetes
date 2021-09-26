package main

import (
	"./vhd"
	"fmt"
	"github.com/codegangsta/cli"
	"github.com/dustin/go-humanize"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func createVHD(file, size string, options vhd.VHDOptions) {

	isize, err := humanize.ParseBytes(size)

	if err != nil {
		panic(err)
	}

	vhd.VHDCreateSparse(uint64(isize), file, options)
	fmt.Printf("File %s (%s) created\n", file, humanize.IBytes(uint64(isize)))
}

func rawToFixed(file string, options *vhd.VHDOptions) {
	f, err := os.OpenFile(file, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		fmt.Printf("Error opening file %s: %s\n", file, err)
		os.Exit(1)
	}
	vhd.RawToFixed(f, options)
	f.Close()
	os.Rename(file, strings.Replace(file, filepath.Ext(file), ".vhd", -1))
}

func vhdInfo(vhdFile string) {

	f, err := os.Open(vhdFile)
	if err != nil {
		fmt.Printf("Error opening file %s: %s\n", vhdFile, err)
		os.Exit(1)
	}
	defer f.Close()

	vhd := vhd.FromFile(f)
	vhd.PrintInfo()
}

func main() {
	app := cli.NewApp()
	app.Version = PKG_VERSION
	app.Name = PKG_NAME
	app.Usage = "Library and tool to manipulate VHD images"
	app.Author = "Sergio Rubio"
	app.Email = "rubiojr@frameos.org"

	app.Commands = []cli.Command{
		{
			Name:  "create",
			Usage: "Create a VHD",
			Action: func(c *cli.Context) {
				if len(c.Args()) != 2 {
					println("Missing command arguments.\n")
					fmt.Printf("Usage: %s create <file-path> <size MiB|GiB|...>\n",
						app.Name)
					os.Exit(1)
				}

				opts := vhd.VHDOptions{}

				tstamp := c.String("timestamp")
				if tstamp != "" {
					itstamp, err := strconv.Atoi(tstamp)
					if err != nil {
						panic(err)
					}
					opts.Timestamp = int64(itstamp)
				}

				uuid := c.String("uuid")
				if uuid != "" {
					opts.UUID = uuid
				}
				createVHD(c.Args()[0], c.Args()[1], opts)
			},
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "uuid",
					Value: "",
					Usage: "Set the UUID of the VHD header",
				},
				cli.StringFlag{
					Name:  "timestamp",
					Value: "",
					Usage: "Set the timestamp of the VHD header (UNIX time format)",
				},
			},
		},
		{
			Name:  "info",
			Usage: "Print VHD info",
			Action: func(c *cli.Context) {
				if len(c.Args()) != 1 {
					println("Missing command arguments.\n")
					fmt.Printf("Usage: %s info <file-path> <size MiB|GiB|...>\n",
						app.Name)
					os.Exit(1)
				}
				vhdInfo(c.Args()[0])
			},
		},
		{
			Name:  "raw2fixed",
			Usage: "Convert a RAW image to a fixed VHD",
			Action: func(c *cli.Context) {
				if len(c.Args()) != 1 {
					println("Missing command arguments.\n")
					fmt.Printf("Usage: %s raw2fixed <file-path>\n",
						app.Name)
					os.Exit(1)
				}

				opts := vhd.VHDOptions{}

				tstamp := c.String("timestamp")
				if tstamp != "" {
					itstamp, err := strconv.Atoi(tstamp)
					if err != nil {
						panic(err)
					}
					opts.Timestamp = int64(itstamp)
				}

				uuid := c.String("uuid")
				if uuid != "" {
					opts.UUID = uuid
				}
				rawToFixed(c.Args()[0], &opts)
			},
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "uuid",
					Value: "",
					Usage: "Set the UUID of the VHD header",
				},
				cli.StringFlag{
					Name:  "timestamp",
					Value: "",
					Usage: "Set the timestamp of the VHD header (UNIX time format)",
				},
			},
		},
	}

	app.Run(os.Args)
}
