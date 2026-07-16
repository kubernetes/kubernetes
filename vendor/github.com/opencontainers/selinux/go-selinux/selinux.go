package selinux

import (
	"errors"
)

const (
	// Enforcing constant indicate SELinux is in enforcing mode
	Enforcing = 1
	// Permissive constant to indicate SELinux is in permissive mode
	Permissive = 0
	// Disabled constant to indicate SELinux is disabled
	Disabled = -1
	// maxCategory is the maximum number of categories used within containers
	maxCategory = 1024
	// DefaultCategoryRange is the upper bound on the category range
	DefaultCategoryRange = uint32(maxCategory)
)

var (
	// ErrMCSAlreadyExists is returned when trying to allocate a duplicate MCS.
	ErrMCSAlreadyExists = errors.New("MCS label already exists")
	// ErrEmptyPath is returned when an empty path has been specified.
	ErrEmptyPath = errors.New("empty path")

	// ErrInvalidLabel is returned when an invalid label is specified.
	ErrInvalidLabel = errors.New("invalid Label")

	// InvalidLabel is returned when an invalid label is specified.
	//
	// Deprecated: use [ErrInvalidLabel].
	InvalidLabel = ErrInvalidLabel

	// ErrIncomparable is returned two levels are not comparable
	ErrIncomparable = errors.New("incomparable levels")
	// ErrLevelSyntax is returned when a sensitivity or category do not have correct syntax in a level
	ErrLevelSyntax = errors.New("invalid level syntax")

	// ErrContextMissing is returned if a requested context is not found in a file.
	ErrContextMissing = errors.New("context does not have a match")
	// ErrVerifierNil is returned when a context verifier function is nil.
	ErrVerifierNil = errors.New("verifier function is nil")

	// ErrNotTGLeader is returned by [SetKeyLabel] if the calling thread
	// is not the thread group leader.
	ErrNotTGLeader = errors.New("calling thread is not the thread group leader")

	// CategoryRange allows the upper bound on the category range to be adjusted
	CategoryRange = DefaultCategoryRange

	privContainerMountLabel string
)

// Context is a representation of the SELinux label broken into 4 parts
type Context map[string]string

// SetDisabled disables SELinux support for the package
func SetDisabled() {
	setDisabled()
}

// GetEnabled returns whether SELinux is currently enabled.
func GetEnabled() bool {
	return getEnabled()
}

// ClassIndex returns the int index for an object class in the loaded policy,
// or -1 and an error
func ClassIndex(class string) (int, error) {
	return classIndex(class)
}

// SetFileLabel sets the SELinux label for this path, following symlinks,
// or returns an error.
func SetFileLabel(fpath string, label string) error {
	return setFileLabel(fpath, label)
}

// LsetFileLabel sets the SELinux label for this path, not following symlinks,
// or returns an error.
func LsetFileLabel(fpath string, label string) error {
	return lSetFileLabel(fpath, label)
}

// FileLabel returns the SELinux label for this path, following symlinks,
// or returns an error.
func FileLabel(fpath string) (string, error) {
	return fileLabel(fpath)
}

// LfileLabel returns the SELinux label for this path, not following symlinks,
// or returns an error.
func LfileLabel(fpath string) (string, error) {
	return lFileLabel(fpath)
}

// SetFSCreateLabel tells the kernel what label to use for all file system objects
// created by this task.
// Set the label to an empty string to return to the default label. Calls to SetFSCreateLabel
// should be wrapped in runtime.LockOSThread()/runtime.UnlockOSThread() until file system
// objects created by this task are finished to guarantee another goroutine does not migrate
// to the current thread before execution is complete.
func SetFSCreateLabel(label string) error {
	return setFSCreateLabel(label)
}

// FSCreateLabel returns the default label the kernel which the kernel is using
// for file system objects created by this task. "" indicates default.
func FSCreateLabel() (string, error) {
	return fsCreateLabel()
}

// CurrentLabel returns the SELinux label of the current process thread, or an error.
func CurrentLabel() (string, error) {
	return currentLabel()
}

// PidLabel returns the SELinux label of the given pid, or an error.
func PidLabel(pid int) (string, error) {
	return pidLabel(pid)
}

// ExecLabel returns the SELinux label that the kernel will use for any programs
// that are executed by the current process thread, or an error.
func ExecLabel() (string, error) {
	return execLabel()
}

// CanonicalizeContext takes a context string and writes it to the kernel
// the function then returns the context that the kernel will use. Use this
// function to check if two contexts are equivalent
func CanonicalizeContext(val string) (string, error) {
	return canonicalizeContext(val)
}

// ComputeCreateContext requests the type transition from source to target for
// class from the kernel.
func ComputeCreateContext(source string, target string, class string) (string, error) {
	return computeCreateContext(source, target, class)
}

// CalculateGlbLub computes the glb (greatest lower bound) and lub (least upper bound)
// of a source and target range.
// The glblub is calculated as the greater of the low sensitivities and
// the lower of the high sensitivities and the and of each category bitset.
func CalculateGlbLub(sourceRange, targetRange string) (string, error) {
	return calculateGlbLub(sourceRange, targetRange)
}

// SetExecLabel sets the SELinux label that the kernel will use for any programs
// that are executed by the current process thread, or an error. Calls to SetExecLabel
// should  be wrapped in runtime.LockOSThread()/runtime.UnlockOSThread() until execution
// of the program is finished to guarantee another goroutine does not migrate to the current
// thread before execution is complete.
func SetExecLabel(label string) error {
	return writeConThreadSelf("attr/exec", label)
}

// SetTaskLabel sets the SELinux label for the current thread, or an error.
// This requires the dyntransition permission. Calls to SetTaskLabel should
// be wrapped in runtime.LockOSThread()/runtime.UnlockOSThread() to guarantee
// the current thread does not run in a new mislabeled thread.
func SetTaskLabel(label string) error {
	return writeConThreadSelf("attr/current", label)
}

// SetSocketLabel takes a process label and tells the kernel to assign the
// label to the next socket that gets created. Calls to SetSocketLabel
// should be wrapped in runtime.LockOSThread()/runtime.UnlockOSThread() until
// the socket is created to guarantee another goroutine does not migrate
// to the current thread before execution is complete.
func SetSocketLabel(label string) error {
	return writeConThreadSelf("attr/sockcreate", label)
}

// SocketLabel retrieves the current socket label setting
func SocketLabel() (string, error) {
	return readConThreadSelf("attr/sockcreate")
}

// PeerLabel retrieves the label of the client on the other side of a socket
func PeerLabel(fd uintptr) (string, error) {
	return peerLabel(fd)
}

// SetKeyLabel takes a process label and tells the kernel to assign the
// label to the next kernel keyring that gets created.
//
// Calls to SetKeyLabel should be wrapped in
// runtime.LockOSThread()/runtime.UnlockOSThread() until the kernel keyring is
// created to guarantee another goroutine does not migrate to the current
// thread before execution is complete.
//
// Only the thread group leader can set key label.
func SetKeyLabel(label string) error {
	return setKeyLabel(label)
}

// KeyLabel retrieves the current kernel keyring label setting
func KeyLabel() (string, error) {
	return keyLabel()
}

// Get returns the Context as a string
func (c Context) Get() string {
	return c.get()
}

// NewContext creates a new Context struct from the specified label
func NewContext(label string) (Context, error) {
	return newContext(label)
}

// ClearLabels clears all reserved labels
func ClearLabels() {
	clearLabels()
}

// ReserveLabel reserves the MLS/MCS level component of the specified label
func ReserveLabel(label string) {
	reserveLabel(label)
}

// MLSEnabled checks if MLS is enabled.
func MLSEnabled() bool {
	return isMLSEnabled()
}

// EnforceMode returns the current SELinux mode Enforcing, Permissive, Disabled
func EnforceMode() int {
	return enforceMode()
}

// SetEnforceMode sets the current SELinux mode Enforcing, Permissive.
// Disabled is not valid, since this needs to be set at boot time.
func SetEnforceMode(mode int) error {
	return setEnforceMode(mode)
}

// DefaultEnforceMode returns the systems default SELinux mode Enforcing,
// Permissive or Disabled. Note this is just the default at boot time.
// EnforceMode tells you the systems current mode.
func DefaultEnforceMode() int {
	return defaultEnforceMode()
}

// ReleaseLabel un-reserves the MLS/MCS Level field of the specified label,
// allowing it to be used by another process.
func ReleaseLabel(label string) {
	releaseLabel(label)
}

// ROFileLabel returns the specified SELinux readonly file label
func ROFileLabel() string {
	return roFileLabel()
}

// KVMContainerLabels returns the default processLabel and mountLabel to be used
// for kvm containers by the calling process.
func KVMContainerLabels() (string, string) {
	return kvmContainerLabels()
}

// InitContainerLabels returns the default processLabel and file labels to be
// used for containers running an init system like systemd by the calling process.
func InitContainerLabels() (string, string) {
	return initContainerLabels()
}

// ContainerLabels returns an allocated processLabel and fileLabel to be used for
// container labeling by the calling process.
func ContainerLabels() (processLabel string, fileLabel string) {
	return containerLabels()
}

// SecurityCheckContext validates that the SELinux label is understood by the kernel
func SecurityCheckContext(val string) error {
	return securityCheckContext(val)
}

// CopyLevel returns a label with the MLS/MCS level from src label replaced on
// the dest label.
func CopyLevel(src, dest string) (string, error) {
	return copyLevel(src, dest)
}

// Chcon changes the fpath file object to the SELinux label.
// If fpath is a directory and recurse is true, then Chcon walks the
// directory tree setting the label.
//
// The fpath itself is guaranteed to be relabeled last.
func Chcon(fpath string, label string, recurse bool) error {
	return chcon(fpath, label, recurse)
}

// DupSecOpt takes an SELinux process label and returns security options that
// can be used to set the SELinux Type and Level for future container processes.
func DupSecOpt(src string) ([]string, error) {
	return dupSecOpt(src)
}

// DisableSecOpt returns a security opt that can be used to disable SELinux
// labeling support for future container processes.
func DisableSecOpt() []string {
	return []string{"disable"}
}

// GetDefaultContextWithLevel gets a single context for the specified SELinux user
// identity that is reachable from the specified scon context. The context is based
// on the per-user /etc/selinux/{SELINUXTYPE}/contexts/users/<username> if it exists,
// and falls back to the global /etc/selinux/{SELINUXTYPE}/contexts/default_contexts
// file.
func GetDefaultContextWithLevel(user, level, scon string) (string, error) {
	return getDefaultContextWithLevel(user, level, scon)
}

// PrivContainerMountLabel returns mount label for privileged containers
func PrivContainerMountLabel() string {
	// Make sure label is initialized.
	_ = label("")
	return privContainerMountLabel
}
