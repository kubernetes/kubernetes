package git_test

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	"gopkg.in/src-d/go-billy.v3/memfs"
)

func ExampleClone() {
	// Filesystem abstraction based on memory
	fs := memfs.New()
	// Git objects storer based on memory
	storer := memory.NewStorage()

	// Clones the repository into the worktree (fs) and storer all the .git
	// content into the storer
	_, _ = git.Clone(storer, fs, &git.CloneOptions{
		URL: "https://github.com/git-fixtures/basic.git",
	})

	// Prints the content of the CHANGELOG file from the cloned repository
	changelog, _ := fs.Open("CHANGELOG")

	io.Copy(os.Stdout, changelog)
	// Output: Initial changelog
}

func ExamplePlainClone() {
	// Tempdir to clone the repository
	dir, err := ioutil.TempDir("", "clone-example")
	if err != nil {
		log.Fatal(err)
	}

	defer os.RemoveAll(dir) // clean up

	// Clones the repository into the given dir, just as a normal git clone does
	_, err = git.PlainClone(dir, false, &git.CloneOptions{
		URL: "https://github.com/git-fixtures/basic.git",
	})

	if err != nil {
		log.Fatal(err)
	}

	// Prints the content of the CHANGELOG file from the cloned repository
	changelog, err := os.Open(filepath.Join(dir, "CHANGELOG"))
	if err != nil {
		log.Fatal(err)
	}

	io.Copy(os.Stdout, changelog)
	// Output: Initial changelog
}

func ExampleRepository_References() {
	r, _ := git.Clone(memory.NewStorage(), nil, &git.CloneOptions{
		URL: "https://github.com/git-fixtures/basic.git",
	})

	// simulating a git show-ref
	refs, _ := r.References()
	refs.ForEach(func(ref *plumbing.Reference) error {
		if ref.Type() == plumbing.HashReference {
			fmt.Println(ref)
		}

		return nil
	})

	// Example Output:
	// 6ecf0ef2c2dffb796033e5a02219af86ec6584e5 refs/remotes/origin/master
	// e8d3ffab552895c19b9fcf7aa264d277cde33881 refs/remotes/origin/branch
	// 6ecf0ef2c2dffb796033e5a02219af86ec6584e5 refs/heads/master

}

func ExampleRepository_CreateRemote() {
	r, _ := git.Init(memory.NewStorage(), nil)

	// Add a new remote, with the default fetch refspec
	_, err := r.CreateRemote(&config.RemoteConfig{
		Name: "example",
		URLs: []string{"https://github.com/git-fixtures/basic.git"},
	})

	if err != nil {
		log.Fatal(err)
	}

	list, err := r.Remotes()
	if err != nil {
		log.Fatal(err)
	}

	for _, r := range list {
		fmt.Println(r)
	}

	// Example Output:
	// example https://github.com/git-fixtures/basic.git (fetch)
	// example https://github.com/git-fixtures/basic.git (push)
}
