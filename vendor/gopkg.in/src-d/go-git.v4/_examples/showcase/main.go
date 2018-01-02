package main

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/plumbing/object"

	. "gopkg.in/src-d/go-git.v4/_examples"
)

// Example of an specific use case:
// - Clone a repository in a specific path
// - Get the HEAD reference
// - Using the HEAD reference, obtain the commit this reference is pointing to
// - Print the commit content
// - Using the commit, iterate all its files and print them
// - Print all the commit history with commit messages, short hash and the
// first line of the commit message
func main() {
	CheckArgs("<url> <path>")
	url := os.Args[1]
	path := os.Args[2]

	// Clone the given repository, creating the remote, the local branches
	// and fetching the objects, exactly as:
	Info("git clone %s %s", url, path)

	r, err := git.PlainClone(path, false, &git.CloneOptions{URL: url})
	CheckIfError(err)

	// Getting the latest commit on the current branch
	Info("git log -1")

	// ... retrieving the branch being pointed by HEAD
	ref, err := r.Head()
	CheckIfError(err)

	// ... retrieving the commit object
	commit, err := r.CommitObject(ref.Hash())
	CheckIfError(err)
	fmt.Println(commit)

	// List the tree from HEAD
	Info("git ls-tree -r HEAD")

	// ... retrieve the tree from the commit
	tree, err := commit.Tree()
	CheckIfError(err)

	// ... get the files iterator and print the file
	tree.Files().ForEach(func(f *object.File) error {
		fmt.Printf("100644 blob %s    %s\n", f.Hash, f.Name)
		return nil
	})

	// List the history of the repository
	Info("git log --oneline")

	commitIter, err := r.Log(&git.LogOptions{From: commit.Hash})
	CheckIfError(err)

	err = commitIter.ForEach(func(c *object.Commit) error {
		hash := c.Hash.String()
		line := strings.Split(c.Message, "\n")
		fmt.Println(hash[:7], line[0])

		return nil
	})
	CheckIfError(err)
}
