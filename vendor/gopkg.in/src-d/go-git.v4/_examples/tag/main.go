package main

import (
	"fmt"
	"os"

	"gopkg.in/src-d/go-git.v4"
	. "gopkg.in/src-d/go-git.v4/_examples"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
)

// Basic example of how to list tags.
func main() {
	CheckArgs("<path>")
	path := os.Args[1]

	// We instance a new repository targeting the given path (the .git folder)
	r, err := git.PlainOpen(path)
	CheckIfError(err)

	// List all tag references, both lightweight tags and annotated tags
	Info("git show-ref --tag")

	tagrefs, err := r.Tags()
	CheckIfError(err)
	err = tagrefs.ForEach(func(t *plumbing.Reference) error {
		fmt.Println(t)
		return nil
	})
	CheckIfError(err)

	// Print each annotated tag object (lightweight tags are not included)
	Info("for t in $(git show-ref --tag); do if [ \"$(git cat-file -t $t)\" = \"tag\" ]; then git cat-file -p $t ; fi; done")

	tags, err := r.TagObjects()
	CheckIfError(err)
	err = tags.ForEach(func(t *object.Tag) error {
		fmt.Println(t)
		return nil
	})
	CheckIfError(err)
}
