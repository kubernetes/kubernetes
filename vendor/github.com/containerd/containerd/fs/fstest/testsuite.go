package fstest

import (
	"context"
	"io/ioutil"
	"os"
	"testing"
)

// TestApplier applies the test context
type TestApplier interface {
	TestContext(context.Context) (context.Context, func(), error)
	Apply(context.Context, Applier) (string, func(), error)
}

// FSSuite runs the path test suite
func FSSuite(t *testing.T, a TestApplier) {
	t.Run("Basic", makeTest(t, a, basicTest))
	t.Run("Deletion", makeTest(t, a, deletionTest))
	t.Run("Update", makeTest(t, a, updateTest))
	t.Run("DirectoryPermission", makeTest(t, a, directoryPermissionsTest))
	t.Run("HardlinkUnmodified", makeTest(t, a, hardlinkUnmodified))
	t.Run("HardlinkBeforeUnmodified", makeTest(t, a, hardlinkBeforeUnmodified))
	t.Run("HardlinkBeforeModified", makeTest(t, a, hardlinkBeforeModified))
}

func makeTest(t *testing.T, ta TestApplier, as []Applier) func(t *testing.T) {
	return func(t *testing.T) {
		ctx, cleanup, err := ta.TestContext(context.Background())
		if err != nil {
			t.Fatalf("Unable to get test context: %+v", err)
		}
		defer cleanup()

		applyDir, err := ioutil.TempDir("", "test-expected-")
		if err != nil {
			t.Fatalf("Unable to make temp directory: %+v", err)
		}
		defer os.RemoveAll(applyDir)

		for i, a := range as {
			testDir, c, err := ta.Apply(ctx, a)
			if err != nil {
				t.Fatalf("Apply failed at %d: %+v", i, err)
			}
			if err := a.Apply(applyDir); err != nil {
				if c != nil {
					c()
				}
				t.Fatalf("Error applying change to apply directory: %+v", err)
			}

			err = CheckDirectoryEqual(applyDir, testDir)
			if c != nil {
				c()
			}
			if err != nil {
				t.Fatalf("Directories not equal at %d (expected <> tested): %+v", i, err)
			}
		}
	}
}

var (
	// baseApplier creates a basic filesystem layout
	// with multiple types of files for basic tests.
	baseApplier = Apply(
		CreateDir("/etc/", 0755),
		CreateFile("/etc/hosts", []byte("127.0.0.1 localhost"), 0644),
		Link("/etc/hosts", "/etc/hosts.allow"),
		CreateDir("/usr/local/lib", 0755),
		CreateFile("/usr/local/lib/libnothing.so", []byte{0x00, 0x00}, 0755),
		Symlink("libnothing.so", "/usr/local/lib/libnothing.so.2"),
		CreateDir("/home", 0755),
		CreateDir("/home/derek", 0700),
	)

	// basicTest covers basic operations
	basicTest = []Applier{
		baseApplier,
		Apply(
			CreateFile("/etc/hosts", []byte("127.0.0.1 localhost.localdomain"), 0644),
			CreateFile("/etc/fstab", []byte("/dev/sda1\t/\text4\tdefaults 1 1\n"), 0600),
			CreateFile("/etc/badfile", []byte(""), 0666),
			CreateFile("/home/derek/.zshrc", []byte("#ZSH is just better\n"), 0640),
		),
		Apply(
			Remove("/etc/badfile"),
			Rename("/home/derek", "/home/notderek"),
		),
		Apply(
			RemoveAll("/usr"),
			Remove("/etc/hosts.allow"),
		),
		Apply(
			RemoveAll("/home"),
			CreateDir("/home/derek", 0700),
			CreateFile("/home/derek/.bashrc", []byte("#not going away\n"), 0640),
			Link("/etc/hosts", "/etc/hosts.allow"),
		),
	}

	// deletionTest covers various deletion scenarios to ensure
	// deletions are properly picked up and applied
	deletionTest = []Applier{
		Apply(
			CreateDir("/test/somedir", 0755),
			CreateDir("/lib", 0700),
			CreateFile("/lib/hidden", []byte{}, 0644),
		),
		Apply(
			CreateFile("/test/a", []byte{}, 0644),
			CreateFile("/test/b", []byte{}, 0644),
			CreateDir("/test/otherdir", 0755),
			CreateFile("/test/otherdir/.empty", []byte{}, 0644),
			RemoveAll("/lib"),
			CreateDir("/lib", 0700),
			CreateFile("/lib/not-hidden", []byte{}, 0644),
		),
		Apply(
			Remove("/test/a"),
			Remove("/test/b"),
			RemoveAll("/test/otherdir"),
			CreateFile("/lib/newfile", []byte{}, 0644),
		),
	}

	// updateTest covers file updates for content and permission
	updateTest = []Applier{
		Apply(
			CreateDir("/d1", 0755),
			CreateDir("/d2", 0700),
			CreateFile("/d1/f1", []byte("something..."), 0644),
			CreateFile("/d1/f2", []byte("else..."), 0644),
			CreateFile("/d1/f3", []byte("entirely..."), 0644),
		),
		Apply(
			CreateFile("/d1/f1", []byte("file content of a different length"), 0664),
			Remove("/d1/f3"),
			CreateFile("/d1/f3", []byte("updated content"), 0664),
			Chmod("/d1/f2", 0766),
			Chmod("/d2", 0777),
		),
	}

	// directoryPermissionsTest covers directory permissions on update
	directoryPermissionsTest = []Applier{
		Apply(
			CreateDir("/d1", 0700),
			CreateDir("/d2", 0751),
			CreateDir("/d3", 0777),
		),
		Apply(
			CreateFile("/d1/f", []byte("irrelevant"), 0644),
			CreateDir("/d1/d", 0700),
			CreateFile("/d1/d/f", []byte("irrelevant"), 0644),
			CreateFile("/d2/f", []byte("irrelevant"), 0644),
			CreateFile("/d3/f", []byte("irrelevant"), 0644),
		),
	}

	hardlinkUnmodified = []Applier{
		baseApplier,
		Apply(
			CreateFile("/etc/hosts", []byte("127.0.0.1 localhost.localdomain"), 0644),
		),
		Apply(
			Link("/etc/hosts", "/etc/hosts.deny"),
		),
	}

	// Hardlink name before with modification
	// Tests link is created for unmodified files when new hardlinked file is seen first
	hardlinkBeforeUnmodified = []Applier{
		baseApplier,
		Apply(
			CreateFile("/etc/hosts", []byte("127.0.0.1 localhost.localdomain"), 0644),
		),
		Apply(
			Link("/etc/hosts", "/etc/before-hosts"),
		),
	}

	// Hardlink name after without modification
	// tests link is created for modified file with new hardlink
	hardlinkBeforeModified = []Applier{
		baseApplier,
		Apply(
			CreateFile("/etc/hosts", []byte("127.0.0.1 localhost.localdomain"), 0644),
		),
		Apply(
			Remove("/etc/hosts"),
			CreateFile("/etc/hosts", []byte("127.0.0.1 localhost"), 0644),
			Link("/etc/hosts", "/etc/before-hosts"),
		),
	}
)
