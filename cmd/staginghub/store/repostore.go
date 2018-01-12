package store

import (
	"fmt"
	"log"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/euforia/go-git-server/repository"
	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
)

type SubdirRepoStore struct {
	repo *git.Repository

	mu          sync.Mutex
	lastHash    plumbing.Hash
	subdirRepos map[string]*repository.Repository
}

// NewSubdirRepoStore instantiates a new repo store for subdirectories of the git worktree at the current working
// path. It create virtual commits just including the subdirectory, without any parents.
func NewSubdirRepoStore() (*SubdirRepoStore, error) {
	repo, err := git.PlainOpen(".")
	if err != nil {
		return nil, fmt.Errorf("failed to open repo at .: %v", err)
	}

	return &SubdirRepoStore{repo: repo, subdirRepos: map[string]*repository.Repository{}}, nil
}

// GetRepo with the given id
func (rs *SubdirRepoStore) GetRepo(id string) (*repository.Repository, error) {
	// get HEAD
	head, err := rs.repo.Head()
	if err != nil {
		return nil, fmt.Errorf("failed to get HEAD of repo at .: %s", err)
	}

	rs.mu.Lock()
	defer rs.mu.Unlock()

	// reset cache if HEAD has changed
	if head.Hash().String() != rs.lastHash.String() {
		rs.subdirRepos = map[string]*repository.Repository{}
		log.Printf("HEAD changed to %v. Clearing caches.", head.Hash())
		rs.lastHash = head.Hash()
	}

	id = path.Clean(id)
	id = strings.TrimSuffix(id, ".git")

	if v, ok := rs.subdirRepos[id]; ok {
		return v, nil
	}

	fs, err := rs.repo.Worktree()
	if err != nil {
		return nil, fmt.Errorf("failed to get git working tree: %v", err)
	}

	if s, err := fs.Filesystem.Stat(id); err != nil || !s.IsDir() {
		return nil, fmt.Errorf("not found: %s", id)
	}

	headCommit, err := rs.repo.CommitObject(head.Hash())
	if err != nil {
		return nil, fmt.Errorf("failed to resolve HEAD: %v", err)
	}

	headTree, err := headCommit.Tree()
	if err != nil {
		return nil, fmt.Errorf("failed to get tree of HEAD: %v", err)
	}

	subdirTree, err := headTree.Tree(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get subtree %s of HEAD: %v", id, err)
	}

	subdirTreeHash := subdirTree.Hash

	now := time.Now()
	sc := &object.Commit{
		Author: object.Signature{
			Name:  "Kubernetes staginghub",
			Email: "noreply@k8s.io",
			When:  now,
		},
		Committer: object.Signature{
			Name:  "Kubernetes staginghub",
			Email: "noreply@k8s.io",
			When:  now,
		},
		Message:  fmt.Sprintf("Subdirectory %s at %s", id, head.Hash()),
		TreeHash: subdirTreeHash,
	}

	subdirCommitEncoded := rs.repo.Storer.NewEncodedObject()
	if err := sc.Encode(subdirCommitEncoded); err != nil {
		return nil, fmt.Errorf("failed to encode subdir %s commit of HEAD: %v", id, err)
	}
	subdirCommitHash, err := rs.repo.Storer.SetEncodedObject(subdirCommitEncoded)
	if err != nil {
		return nil, fmt.Errorf("failed to store subdir %s commit of HEAD: %v", id, err)
	}

	subdirRepo := repository.NewRepository(id)
	subdirRepo.Refs.Head.Hash = subdirCommitHash
	subdirRepo.Refs.Heads["master"] = subdirCommitHash
	rs.subdirRepos[id] = subdirRepo
	return subdirRepo, nil
}

// CreateRepo with the given repo data
func (rs *SubdirRepoStore) CreateRepo(repo *repository.Repository) error {
	return fmt.Errorf("cannot create repo %q", repo.ID)
}

// UpdateRepo with the given data
func (rs *SubdirRepoStore) UpdateRepo(repo *repository.Repository) error {
	return fmt.Errorf("cannot update repo %q", repo.ID)
}

// RemoveRepo with the given id
func (rs *SubdirRepoStore) RemoveRepo(id string) error {
	return fmt.Errorf("cannot remove repo %q", id)
}
