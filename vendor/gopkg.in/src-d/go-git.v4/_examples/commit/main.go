package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/src-d/go-git.v4"
	. "gopkg.in/src-d/go-git.v4/_examples"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
)

// Basic example of how to commit changes to the current branch to an existant
// repository.
func main() {
	CheckArgs("<directory>")
	directory := os.Args[1]

	// Opens an already existant repository.
	r, err := git.PlainOpen(directory)
	CheckIfError(err)

	w, err := r.Worktree()
	CheckIfError(err)

	// ... we need a file to commit so let's create a new file inside of the
	// worktree of the project using the go standard library.
	Info("echo \"hellow world!\" > example-git-file")
	filename := filepath.Join(directory, "example-git-file")
	err = ioutil.WriteFile(filename, []byte("hello world!"), 0644)
	CheckIfError(err)

	// Adds the new file to the staging area.
	Info("git add example-git-file")
	_, err = w.Add("example-git-file")
	CheckIfError(err)

	// We can verify the current status of the worktree using the method Status.
	Info("git status --porcelain")
	status, err := w.Status()
	CheckIfError(err)

	fmt.Println(status)

	// Commits the current staging are to the repository, with the new file
	// just created. We should provide the object.Signature of Author of the
	// commit.
	Info("git commit -m \"example go-git commit\"")
	commit, err := w.Commit("example go-git commit", &git.CommitOptions{
		Author: &object.Signature{
			Name:  "John Doe",
			Email: "john@doe.org",
			When:  time.Now(),
		},
	})

	CheckIfError(err)

	// Prints the current HEAD to verify that all worked well.
	Info("git show -s")
	obj, err := r.CommitObject(commit)
	CheckIfError(err)

	fmt.Println(obj)
}
