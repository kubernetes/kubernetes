package git

import (
	"context"
	"errors"
	"fmt"
	"io"
	stdioutil "io/ioutil"
	"os"
	"path/filepath"

	"gopkg.in/src-d/go-billy.v3/util"
	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/format/index"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie"

	"gopkg.in/src-d/go-billy.v3"
)

var (
	ErrWorktreeNotClean  = errors.New("worktree is not clean")
	ErrSubmoduleNotFound = errors.New("submodule not found")
	ErrUnstaggedChanges  = errors.New("worktree contains unstagged changes")
)

// Worktree represents a git worktree.
type Worktree struct {
	// Filesystem underlying filesystem.
	Filesystem billy.Filesystem

	r *Repository
}

// Pull incorporates changes from a remote repository into the current branch.
// Returns nil if the operation is successful, NoErrAlreadyUpToDate if there are
// no changes to be fetched, or an error.
//
// Pull only supports merges where the can be resolved as a fast-forward.
func (w *Worktree) Pull(o *PullOptions) error {
	return w.PullContext(context.Background(), o)
}

// PullContext incorporates changes from a remote repository into the current
// branch. Returns nil if the operation is successful, NoErrAlreadyUpToDate if
// there are no changes to be fetched, or an error.
//
// Pull only supports merges where the can be resolved as a fast-forward.
//
// The provided Context must be non-nil. If the context expires before the
// operation is complete, an error is returned. The context only affects to the
// transport operations.
func (w *Worktree) PullContext(ctx context.Context, o *PullOptions) error {
	if err := o.Validate(); err != nil {
		return err
	}

	remote, err := w.r.Remote(o.RemoteName)
	if err != nil {
		return err
	}

	fetchHead, err := remote.fetch(ctx, &FetchOptions{
		RemoteName: o.RemoteName,
		Depth:      o.Depth,
		Auth:       o.Auth,
		Progress:   o.Progress,
	})

	updated := true
	if err == NoErrAlreadyUpToDate {
		updated = false
	} else if err != nil {
		return err
	}

	ref, err := storer.ResolveReference(fetchHead, o.ReferenceName)
	if err != nil {
		return err
	}

	head, err := w.r.Head()
	if err == nil {
		if !updated && head.Hash() == ref.Hash() {
			return NoErrAlreadyUpToDate
		}

		ff, err := isFastForward(w.r.Storer, head.Hash(), ref.Hash())
		if err != nil {
			return err
		}

		if !ff {
			return fmt.Errorf("non-fast-forward update")
		}
	}

	if err != nil && err != plumbing.ErrReferenceNotFound {
		return err
	}

	if err := w.updateHEAD(ref.Hash()); err != nil {
		return err
	}

	if err := w.Reset(&ResetOptions{
		Mode:   MergeReset,
		Commit: ref.Hash(),
	}); err != nil {
		return err
	}

	if o.RecurseSubmodules != NoRecurseSubmodules {
		return w.updateSubmodules(&SubmoduleUpdateOptions{
			RecurseSubmodules: o.RecurseSubmodules,
			Auth:              o.Auth,
		})
	}

	return nil
}

func (w *Worktree) updateSubmodules(o *SubmoduleUpdateOptions) error {
	s, err := w.Submodules()
	if err != nil {
		return err
	}
	o.Init = true
	return s.Update(o)
}

// Checkout switch branches or restore working tree files.
func (w *Worktree) Checkout(opts *CheckoutOptions) error {
	if err := opts.Validate(); err != nil {
		return err
	}

	if opts.Create {
		if err := w.createBranch(opts); err != nil {
			return err
		}
	}

	if !opts.Force {
		unstaged, err := w.containsUnstagedChanges()
		if err != nil {
			return err
		}

		if unstaged {
			return ErrUnstaggedChanges
		}
	}

	c, err := w.getCommitFromCheckoutOptions(opts)
	if err != nil {
		return err
	}

	ro := &ResetOptions{Commit: c, Mode: MergeReset}
	if opts.Force {
		ro.Mode = HardReset
	}

	if !opts.Hash.IsZero() && !opts.Create {
		err = w.setHEADToCommit(opts.Hash)
	} else {
		err = w.setHEADToBranch(opts.Branch, c)
	}

	if err != nil {
		return err
	}

	return w.Reset(ro)
}
func (w *Worktree) createBranch(opts *CheckoutOptions) error {
	_, err := w.r.Storer.Reference(opts.Branch)
	if err == nil {
		return fmt.Errorf("a branch named %q already exists", opts.Branch)
	}

	if err != plumbing.ErrReferenceNotFound {
		return err
	}

	if opts.Hash.IsZero() {
		ref, err := w.r.Head()
		if err != nil {
			return err
		}

		opts.Hash = ref.Hash()
	}

	return w.r.Storer.SetReference(
		plumbing.NewHashReference(opts.Branch, opts.Hash),
	)
}

func (w *Worktree) getCommitFromCheckoutOptions(opts *CheckoutOptions) (plumbing.Hash, error) {
	if !opts.Hash.IsZero() {
		return opts.Hash, nil
	}

	b, err := w.r.Reference(opts.Branch, true)
	if err != nil {
		return plumbing.ZeroHash, err
	}

	if !b.Name().IsTag() {
		return b.Hash(), nil
	}

	o, err := w.r.Object(plumbing.AnyObject, b.Hash())
	if err != nil {
		return plumbing.ZeroHash, err
	}

	switch o := o.(type) {
	case *object.Tag:
		if o.TargetType != plumbing.CommitObject {
			return plumbing.ZeroHash, fmt.Errorf("unsupported tag object target %q", o.TargetType)
		}

		return o.Target, nil
	case *object.Commit:
		return o.Hash, nil
	}

	return plumbing.ZeroHash, fmt.Errorf("unsupported tag target %q", o.Type())
}

func (w *Worktree) setHEADToCommit(commit plumbing.Hash) error {
	head := plumbing.NewHashReference(plumbing.HEAD, commit)
	return w.r.Storer.SetReference(head)
}

func (w *Worktree) setHEADToBranch(branch plumbing.ReferenceName, commit plumbing.Hash) error {
	target, err := w.r.Storer.Reference(branch)
	if err != nil {
		return err
	}

	var head *plumbing.Reference
	if target.Name().IsBranch() {
		head = plumbing.NewSymbolicReference(plumbing.HEAD, target.Name())
	} else {
		head = plumbing.NewHashReference(plumbing.HEAD, commit)
	}

	return w.r.Storer.SetReference(head)
}

// Reset the worktree to a specified state.
func (w *Worktree) Reset(opts *ResetOptions) error {
	if err := opts.Validate(w.r); err != nil {
		return err
	}

	if opts.Mode == MergeReset {
		unstaged, err := w.containsUnstagedChanges()
		if err != nil {
			return err
		}

		if unstaged {
			return ErrUnstaggedChanges
		}
	}

	if err := w.setHEADCommit(opts.Commit); err != nil {
		return err
	}

	if opts.Mode == SoftReset {
		return nil
	}

	t, err := w.getTreeFromCommitHash(opts.Commit)
	if err != nil {
		return err
	}

	if opts.Mode == MixedReset || opts.Mode == MergeReset || opts.Mode == HardReset {
		if err := w.resetIndex(t); err != nil {
			return err
		}
	}

	if opts.Mode == MergeReset || opts.Mode == HardReset {
		if err := w.resetWorktree(t); err != nil {
			return err
		}
	}

	return nil
}

func (w *Worktree) resetIndex(t *object.Tree) error {
	idx, err := w.r.Storer.Index()
	if err != nil {
		return err
	}

	changes, err := w.diffTreeWithStaging(t, true)
	if err != nil {
		return err
	}

	for _, ch := range changes {
		a, err := ch.Action()
		if err != nil {
			return err
		}

		var name string
		var e *object.TreeEntry

		switch a {
		case merkletrie.Modify, merkletrie.Insert:
			name = ch.To.String()
			e, err = t.FindEntry(name)
			if err != nil {
				return err
			}
		case merkletrie.Delete:
			name = ch.From.String()
		}

		_, _ = idx.Remove(name)
		if e == nil {
			continue
		}

		idx.Entries = append(idx.Entries, &index.Entry{
			Name: name,
			Hash: e.Hash,
			Mode: e.Mode,
		})

	}

	return w.r.Storer.SetIndex(idx)
}

func (w *Worktree) resetWorktree(t *object.Tree) error {
	changes, err := w.diffStagingWithWorktree(true)
	if err != nil {
		return err
	}

	idx, err := w.r.Storer.Index()
	if err != nil {
		return err
	}

	for _, ch := range changes {
		if err := w.checkoutChange(ch, t, idx); err != nil {
			return err
		}
	}

	return w.r.Storer.SetIndex(idx)
}

func (w *Worktree) checkoutChange(ch merkletrie.Change, t *object.Tree, idx *index.Index) error {
	a, err := ch.Action()
	if err != nil {
		return err
	}

	var e *object.TreeEntry
	var name string
	var isSubmodule bool

	switch a {
	case merkletrie.Modify, merkletrie.Insert:
		name = ch.To.String()
		e, err = t.FindEntry(name)
		if err != nil {
			return err
		}

		isSubmodule = e.Mode == filemode.Submodule
	case merkletrie.Delete:
		return rmFileAndDirIfEmpty(w.Filesystem, ch.From.String())
	}

	if isSubmodule {
		return w.checkoutChangeSubmodule(name, a, e, idx)
	}

	return w.checkoutChangeRegularFile(name, a, t, e, idx)
}

func (w *Worktree) containsUnstagedChanges() (bool, error) {
	ch, err := w.diffStagingWithWorktree(false)
	if err != nil {
		return false, err
	}

	for _, c := range ch {
		a, err := c.Action()
		if err != nil {
			return false, err
		}

		if a == merkletrie.Insert {
			continue
		}

		return true, nil
	}

	return false, nil
}

func (w *Worktree) setHEADCommit(commit plumbing.Hash) error {
	head, err := w.r.Reference(plumbing.HEAD, false)
	if err != nil {
		return err
	}

	if head.Type() == plumbing.HashReference {
		head = plumbing.NewHashReference(plumbing.HEAD, commit)
		return w.r.Storer.SetReference(head)
	}

	branch, err := w.r.Reference(head.Target(), false)
	if err != nil {
		return err
	}

	if !branch.Name().IsBranch() {
		return fmt.Errorf("invalid HEAD target should be a branch, found %s", branch.Type())
	}

	branch = plumbing.NewHashReference(branch.Name(), commit)
	return w.r.Storer.SetReference(branch)
}

func (w *Worktree) checkoutChangeSubmodule(name string,
	a merkletrie.Action,
	e *object.TreeEntry,
	idx *index.Index,
) error {
	switch a {
	case merkletrie.Modify:
		sub, err := w.Submodule(name)
		if err != nil {
			return err
		}

		if !sub.initialized {
			return nil
		}

		return w.addIndexFromTreeEntry(name, e, idx)
	case merkletrie.Insert:
		mode, err := e.Mode.ToOSFileMode()
		if err != nil {
			return err
		}

		if err := w.Filesystem.MkdirAll(name, mode); err != nil {
			return err
		}

		return w.addIndexFromTreeEntry(name, e, idx)
	}

	return nil
}

func (w *Worktree) checkoutChangeRegularFile(name string,
	a merkletrie.Action,
	t *object.Tree,
	e *object.TreeEntry,
	idx *index.Index,
) error {
	switch a {
	case merkletrie.Modify:
		_, _ = idx.Remove(name)

		// to apply perm changes the file is deleted, billy doesn't implement
		// chmod
		if err := w.Filesystem.Remove(name); err != nil {
			return err
		}

		fallthrough
	case merkletrie.Insert:
		f, err := t.File(name)
		if err != nil {
			return err
		}

		if err := w.checkoutFile(f); err != nil {
			return err
		}

		return w.addIndexFromFile(name, e.Hash, idx)
	}

	return nil
}

func (w *Worktree) checkoutFile(f *object.File) (err error) {
	mode, err := f.Mode.ToOSFileMode()
	if err != nil {
		return
	}

	if mode&os.ModeSymlink != 0 {
		return w.checkoutFileSymlink(f)
	}

	from, err := f.Reader()
	if err != nil {
		return
	}

	defer ioutil.CheckClose(from, &err)

	to, err := w.Filesystem.OpenFile(f.Name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode.Perm())
	if err != nil {
		return
	}

	defer ioutil.CheckClose(to, &err)

	_, err = io.Copy(to, from)
	return
}

func (w *Worktree) checkoutFileSymlink(f *object.File) (err error) {
	from, err := f.Reader()
	if err != nil {
		return
	}

	defer ioutil.CheckClose(from, &err)

	bytes, err := stdioutil.ReadAll(from)
	if err != nil {
		return
	}

	err = w.Filesystem.Symlink(string(bytes), f.Name)
	return
}

func (w *Worktree) addIndexFromTreeEntry(name string, f *object.TreeEntry, idx *index.Index) error {
	_, _ = idx.Remove(name)
	idx.Entries = append(idx.Entries, &index.Entry{
		Hash: f.Hash,
		Name: name,
		Mode: filemode.Submodule,
	})

	return nil
}

func (w *Worktree) addIndexFromFile(name string, h plumbing.Hash, idx *index.Index) error {
	_, _ = idx.Remove(name)
	fi, err := w.Filesystem.Lstat(name)
	if err != nil {
		return err
	}

	mode, err := filemode.NewFromOSFileMode(fi.Mode())
	if err != nil {
		return err
	}

	e := &index.Entry{
		Hash:       h,
		Name:       name,
		Mode:       mode,
		ModifiedAt: fi.ModTime(),
		Size:       uint32(fi.Size()),
	}

	// if the FileInfo.Sys() comes from os the ctime, dev, inode, uid and gid
	// can be retrieved, otherwise this doesn't apply
	if fillSystemInfo != nil {
		fillSystemInfo(e, fi.Sys())
	}

	idx.Entries = append(idx.Entries, e)
	return nil
}

func (w *Worktree) getTreeFromCommitHash(commit plumbing.Hash) (*object.Tree, error) {
	c, err := w.r.CommitObject(commit)
	if err != nil {
		return nil, err
	}

	return c.Tree()
}

func (w *Worktree) initializeIndex() error {
	return w.r.Storer.SetIndex(&index.Index{Version: 2})
}

var fillSystemInfo func(e *index.Entry, sys interface{})

const gitmodulesFile = ".gitmodules"

// Submodule returns the submodule with the given name
func (w *Worktree) Submodule(name string) (*Submodule, error) {
	l, err := w.Submodules()
	if err != nil {
		return nil, err
	}

	for _, m := range l {
		if m.Config().Name == name {
			return m, nil
		}
	}

	return nil, ErrSubmoduleNotFound
}

// Submodules returns all the available submodules
func (w *Worktree) Submodules() (Submodules, error) {
	l := make(Submodules, 0)
	m, err := w.readGitmodulesFile()
	if err != nil || m == nil {
		return l, err
	}

	c, err := w.r.Config()
	if err != nil {
		return nil, err
	}

	for _, s := range m.Submodules {
		l = append(l, w.newSubmodule(s, c.Submodules[s.Name]))
	}

	return l, nil
}

func (w *Worktree) newSubmodule(fromModules, fromConfig *config.Submodule) *Submodule {
	m := &Submodule{w: w}
	m.initialized = fromConfig != nil

	if !m.initialized {
		m.c = fromModules
		return m
	}

	m.c = fromConfig
	m.c.Path = fromModules.Path
	return m
}

func (w *Worktree) readGitmodulesFile() (*config.Modules, error) {
	f, err := w.Filesystem.Open(gitmodulesFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}

		return nil, err
	}

	defer f.Close()
	input, err := stdioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	m := config.NewModules()
	return m, m.Unmarshal(input)
}

func rmFileAndDirIfEmpty(fs billy.Filesystem, name string) error {
	if err := util.RemoveAll(fs, name); err != nil {
		return err
	}

	path := filepath.Dir(name)
	files, err := fs.ReadDir(path)
	if err != nil {
		return err
	}

	if len(files) == 0 {
		fs.Remove(path)
	}

	return nil
}
