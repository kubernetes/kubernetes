package main

import (
	"os"

	"gopkg.in/src-d/go-git.v4"
	. "gopkg.in/src-d/go-git.v4/_examples"
)

// Example of how to open a repository in a specific path, and push to
// its default remote (origin).
func main() {
	CheckArgs("<repository-path>")
	path := os.Args[1]

	r, err := git.PlainOpen(path)
	CheckIfError(err)

	Info("git push")
	// push using default options
	err = r.Push(&git.PushOptions{})
	CheckIfError(err)
}
