package git

import (
	"context"
	"errors"
	"fmt"
	"io"
	stdioutil "io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/go-git/go-git/v5/config"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/filemode"
	"github.com/go-git/go-git/v5/plumbing/format/gitignore"
	"github.com/go-git/go-git/v5/plumbing/format/index"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/utils/ioutil"
	"github.com/go-git/go-git/v5/utils/merkletrie"

	"github.com/go-git/go-billy/v5"
	"github.com/go-git/go-billy/v5/util"
)

var (
	ErrWorktreeNotClean     = errors.New("worktree is not clean")
	ErrSubmoduleNotFound    = errors.New("submodule not found")
	ErrUnstagedChanges      = errors.New("worktree contains unstaged changes")
	ErrGitModulesSymlink    = errors.New(gitmodulesFile + " is a symlink")
	ErrNonFastForwardUpdate = errors.New("non-fast-forward update")
)

// Worktree represents a git worktree.
type Worktree struct {
	// Filesystem underlying filesystem.
	Filesystem billy.Filesystem
	// External excludes not found in the repository .gitignore
	Excludes []gitignore.Pattern

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
		Force:      o.Force,
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
		headAheadOfRef, err := isFastForward(w.r.Storer, ref.Hash(), head.Hash())
		if err != nil {
			return err
		}

		if !updated && headAheadOfRef {
			return NoErrAlreadyUpToDate
		}

		ff, err := isFastForward(w.r.Storer, head.Hash(), ref.Hash())
		if err != nil {
			return err
		}

		if !ff {
			return ErrNonFastForwardUpdate
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

	c, err := w.getCommitFromCheckoutOptions(opts)
	if err != nil {
		return err
	}

	ro := &ResetOptions{Commit: c, Mode: MergeReset}
	if opts.Force {
		ro.Mode = HardReset
	} else if opts.Keep {
		ro.Mode = SoftReset
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
			return ErrUnstagedChanges
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
	b := newIndexBuilder(idx)

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

		b.Remove(name)
		if e == nil {
			continue
		}

		b.Add(&index.Entry{
			Name: name,
			Hash: e.Hash,
			Mode: e.Mode,
		})

	}

	b.Write(idx)
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
	b := newIndexBuilder(idx)

	for _, ch := range changes {
		if err := w.checkoutChange(ch, t, b); err != nil {
			return err
		}
	}

	b.Write(idx)
	return w.r.Storer.SetIndex(idx)
}

func (w *Worktree) checkoutChange(ch merkletrie.Change, t *object.Tree, idx *indexBuilder) error {
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
	idx *indexBuilder,
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
	idx *indexBuilder,
) error {
	switch a {
	case merkletrie.Modify:
		idx.Remove(name)

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

var copyBufferPool = sync.Pool{
	New: func() interface{} {
		return make([]byte, 32*1024)
	},
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
	buf := copyBufferPool.Get().([]byte)
	_, err = io.CopyBuffer(to, from, buf)
	copyBufferPool.Put(buf)
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

	// On windows, this might fail.
	// Follow Git on Windows behavior by writing the link as it is.
	if err != nil && isSymlinkWindowsNonAdmin(err) {
		mode, _ := f.Mode.ToOSFileMode()

		to, err := w.Filesystem.OpenFile(f.Name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, mode.Perm())
		if err != nil {
			return err
		}

		defer ioutil.CheckClose(to, &err)

		_, err = to.Write(bytes)
		return err
	}
	return
}

func (w *Worktree) addIndexFromTreeEntry(name string, f *object.TreeEntry, idx *indexBuilder) error {
	idx.Remove(name)
	idx.Add(&index.Entry{
		Hash: f.Hash,
		Name: name,
		Mode: filemode.Submodule,
	})
	return nil
}

func (w *Worktree) addIndexFromFile(name string, h plumbing.Hash, idx *indexBuilder) error {
	idx.Remove(name)
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
	idx.Add(e)
	return nil
}

func (w *Worktree) getTreeFromCommitHash(commit plumbing.Hash) (*object.Tree, error) {
	c, err := w.r.CommitObject(commit)
	if err != nil {
		return nil, err
	}

	return c.Tree()
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

func (w *Worktree) isSymlink(path string) bool {
	if s, err := w.Filesystem.Lstat(path); err == nil {
		return s.Mode()&os.ModeSymlink != 0
	}
	return false
}

func (w *Worktree) readGitmodulesFile() (*config.Modules, error) {
	if w.isSymlink(gitmodulesFile) {
		return nil, ErrGitModulesSymlink
	}

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

// Clean the worktree by removing untracked files.
// An empty dir could be removed - this is what  `git clean -f -d .` does.
func (w *Worktree) Clean(opts *CleanOptions) error {
	s, err := w.Status()
	if err != nil {
		return err
	}

	root := ""
	files, err := w.Filesystem.ReadDir(root)
	if err != nil {
		return err
	}
	return w.doClean(s, opts, root, files)
}

func (w *Worktree) doClean(status Status, opts *CleanOptions, dir string, files []os.FileInfo) error {
	for _, fi := range files {
		if fi.Name() == GitDirName {
			continue
		}

		// relative path under the root
		path := filepath.Join(dir, fi.Name())
		if fi.IsDir() {
			if !opts.Dir {
				continue
			}

			subfiles, err := w.Filesystem.ReadDir(path)
			if err != nil {
				return err
			}
			err = w.doClean(status, opts, path, subfiles)
			if err != nil {
				return err
			}
		} else {
			if status.IsUntracked(path) {
				if err := w.Filesystem.Remove(path); err != nil {
					return err
				}
			}
		}
	}

	if opts.Dir {
		return doCleanDirectories(w.Filesystem, dir)
	}
	return nil
}

// GrepResult is structure of a grep result.
type GrepResult struct {
	// FileName is the name of file which contains match.
	FileName string
	// LineNumber is the line number of a file at which a match was found.
	LineNumber int
	// Content is the content of the file at the matching line.
	Content string
	// TreeName is the name of the tree (reference name/commit hash) at
	// which the match was performed.
	TreeName string
}

func (gr GrepResult) String() string {
	return fmt.Sprintf("%s:%s:%d:%s", gr.TreeName, gr.FileName, gr.LineNumber, gr.Content)
}

// Grep performs grep on a worktree.
func (w *Worktree) Grep(opts *GrepOptions) ([]GrepResult, error) {
	if err := opts.Validate(w); err != nil {
		return nil, err
	}

	// Obtain commit hash from options (CommitHash or ReferenceName).
	var commitHash plumbing.Hash
	// treeName contains the value of TreeName in GrepResult.
	var treeName string

	if opts.ReferenceName != "" {
		ref, err := w.r.Reference(opts.ReferenceName, true)
		if err != nil {
			return nil, err
		}
		commitHash = ref.Hash()
		treeName = opts.ReferenceName.String()
	} else if !opts.CommitHash.IsZero() {
		commitHash = opts.CommitHash
		treeName = opts.CommitHash.String()
	}

	// Obtain a tree from the commit hash and get a tracked files iterator from
	// the tree.
	tree, err := w.getTreeFromCommitHash(commitHash)
	if err != nil {
		return nil, err
	}
	fileiter := tree.Files()

	return findMatchInFiles(fileiter, treeName, opts)
}

// findMatchInFiles takes a FileIter, worktree name and GrepOptions, and
// returns a slice of GrepResult containing the result of regex pattern matching
// in content of all the files.
func findMatchInFiles(fileiter *object.FileIter, treeName string, opts *GrepOptions) ([]GrepResult, error) {
	var results []GrepResult

	err := fileiter.ForEach(func(file *object.File) error {
		var fileInPathSpec bool

		// When no pathspecs are provided, search all the files.
		if len(opts.PathSpecs) == 0 {
			fileInPathSpec = true
		}

		// Check if the file name matches with the pathspec. Break out of the
		// loop once a match is found.
		for _, pathSpec := range opts.PathSpecs {
			if pathSpec != nil && pathSpec.MatchString(file.Name) {
				fileInPathSpec = true
				break
			}
		}

		// If the file does not match with any of the pathspec, skip it.
		if !fileInPathSpec {
			return nil
		}

		grepResults, err := findMatchInFile(file, treeName, opts)
		if err != nil {
			return err
		}
		results = append(results, grepResults...)

		return nil
	})

	return results, err
}

// findMatchInFile takes a single File, worktree name and GrepOptions,
// and returns a slice of GrepResult containing the result of regex pattern
// matching in the given file.
func findMatchInFile(file *object.File, treeName string, opts *GrepOptions) ([]GrepResult, error) {
	var grepResults []GrepResult

	content, err := file.Contents()
	if err != nil {
		return grepResults, err
	}

	// Split the file content and parse line-by-line.
	contentByLine := strings.Split(content, "\n")
	for lineNum, cnt := range contentByLine {
		addToResult := false

		// Match the patterns and content. Break out of the loop once a
		// match is found.
		for _, pattern := range opts.Patterns {
			if pattern != nil && pattern.MatchString(cnt) {
				// Add to result only if invert match is not enabled.
				if !opts.InvertMatch {
					addToResult = true
					break
				}
			} else if opts.InvertMatch {
				// If matching fails, and invert match is enabled, add to
				// results.
				addToResult = true
				break
			}
		}

		if addToResult {
			grepResults = append(grepResults, GrepResult{
				FileName:   file.Name,
				LineNumber: lineNum + 1,
				Content:    cnt,
				TreeName:   treeName,
			})
		}
	}

	return grepResults, nil
}

func rmFileAndDirIfEmpty(fs billy.Filesystem, name string) error {
	if err := util.RemoveAll(fs, name); err != nil {
		return err
	}

	dir := filepath.Dir(name)
	return doCleanDirectories(fs, dir)
}

// doCleanDirectories removes empty subdirs (without files)
func doCleanDirectories(fs billy.Filesystem, dir string) error {
	files, err := fs.ReadDir(dir)
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fs.Remove(dir)
	}
	return nil
}

type indexBuilder struct {
	entries map[string]*index.Entry
}

func newIndexBuilder(idx *index.Index) *indexBuilder {
	entries := make(map[string]*index.Entry, len(idx.Entries))
	for _, e := range idx.Entries {
		entries[e.Name] = e
	}
	return &indexBuilder{
		entries: entries,
	}
}

func (b *indexBuilder) Write(idx *index.Index) {
	idx.Entries = idx.Entries[:0]
	for _, e := range b.entries {
		idx.Entries = append(idx.Entries, e)
	}
}

func (b *indexBuilder) Add(e *index.Entry) {
	b.entries[e.Name] = e
}

func (b *indexBuilder) Remove(name string) {
	delete(b.entries, filepath.ToSlash(name))
}
