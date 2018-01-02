package main

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/src-d/go-git.v4/plumbing/transport/file"
)

type CmdUploadPack struct {
	cmd

	Args struct {
		GitDir string `positional-arg-name:"git-dir" required:"true"`
	} `positional-args:"yes"`
}

func (CmdUploadPack) Usage() string {
	//TODO: usage: git upload-pack [--strict] [--timeout=<n>] <dir>
	//TODO: git-upload-pack returns error code 129 if arguments are invalid.
	return fmt.Sprintf("usage: %s <git-dir>", os.Args[0])
}

func (c *CmdUploadPack) Execute(args []string) error {
	gitDir, err := filepath.Abs(c.Args.GitDir)
	if err != nil {
		return err
	}

	if err := file.ServeUploadPack(gitDir); err != nil {
		fmt.Fprintln(os.Stderr, "ERR:", err)
		os.Exit(128)
	}

	return nil
}
