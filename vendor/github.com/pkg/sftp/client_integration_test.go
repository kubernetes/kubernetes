package sftp

// sftp integration tests
// enable with -integration

import (
	"crypto/sha1"
	"flag"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"os/user"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"testing"
	"testing/quick"
	"time"

	"github.com/kr/fs"
)

const (
	READONLY                = true
	READWRITE               = false
	NO_DELAY  time.Duration = 0

	debuglevel = "ERROR" // set to "DEBUG" for debugging
)

var testServerImpl = flag.Bool("testserver", false, "perform integration tests against sftp package server instance")
var testIntegration = flag.Bool("integration", false, "perform integration tests against sftp server process")
var testSftp = flag.String("sftp", sftpServer, "location of the sftp server binary")

type delayedWrite struct {
	t time.Time
	b []byte
}

// delayedWriter wraps a writer and artificially delays the write. This is
// meant to mimic connections with various latencies. Error's returned from the
// underlying writer will panic so this should only be used over reliable
// connections.
type delayedWriter struct {
	w      io.WriteCloser
	ch     chan delayedWrite
	closed chan struct{}
}

func newDelayedWriter(w io.WriteCloser, delay time.Duration) io.WriteCloser {
	ch := make(chan delayedWrite, 128)
	closed := make(chan struct{})
	go func() {
		for writeMsg := range ch {
			time.Sleep(writeMsg.t.Add(delay).Sub(time.Now()))
			n, err := w.Write(writeMsg.b)
			if err != nil {
				panic("write error")
			}
			if n < len(writeMsg.b) {
				panic("showrt write")
			}
		}
		w.Close()
		close(closed)
	}()
	return delayedWriter{w: w, ch: ch, closed: closed}
}

func (w delayedWriter) Write(b []byte) (int, error) {
	bcopy := make([]byte, len(b))
	copy(bcopy, b)
	w.ch <- delayedWrite{t: time.Now(), b: bcopy}
	return len(b), nil
}

func (w delayedWriter) Close() error {
	close(w.ch)
	<-w.closed
	return nil
}

// netPipe provides a pair of io.ReadWriteClosers connected to each other.
// The functions is identical to os.Pipe with the exception that netPipe
// provides the Read/Close guarentees that os.File derrived pipes do not.
func netPipe(t testing.TB) (io.ReadWriteCloser, io.ReadWriteCloser) {
	type result struct {
		net.Conn
		error
	}

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	ch := make(chan result, 1)
	go func() {
		conn, err := l.Accept()
		ch <- result{conn, err}
		err = l.Close()
		if err != nil {
			t.Error(err)
		}
	}()
	c1, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		l.Close() // might cause another in the listening goroutine, but too bad
		t.Fatal(err)
	}
	r := <-ch
	if r.error != nil {
		t.Fatal(err)
	}
	return c1, r.Conn
}

func testClientGoSvr(t testing.TB, readonly bool, delay time.Duration) (*Client, *exec.Cmd) {
	c1, c2 := netPipe(t)

	options := []ServerOption{WithDebug(os.Stderr)}
	if readonly {
		options = append(options, ReadOnly())
	}

	server, err := NewServer(c1, options...)
	if err != nil {
		t.Fatal(err)
	}
	go server.Serve()

	var ctx io.WriteCloser = c2
	if delay > NO_DELAY {
		ctx = newDelayedWriter(ctx, delay)
	}

	client, err := NewClientPipe(c2, ctx)
	if err != nil {
		t.Fatal(err)
	}

	// dummy command...
	return client, exec.Command("true")
}

// testClient returns a *Client connected to a localy running sftp-server
// the *exec.Cmd returned must be defer Wait'd.
func testClient(t testing.TB, readonly bool, delay time.Duration) (*Client, *exec.Cmd) {
	if !*testIntegration {
		t.Skip("skipping intergration test")
	}

	if *testServerImpl {
		return testClientGoSvr(t, readonly, delay)
	}

	cmd := exec.Command(*testSftp, "-e", "-R", "-l", debuglevel) // log to stderr, read only
	if !readonly {
		cmd = exec.Command(*testSftp, "-e", "-l", debuglevel) // log to stderr
	}
	cmd.Stderr = os.Stdout
	pw, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if delay > NO_DELAY {
		pw = newDelayedWriter(pw, delay)
	}
	pr, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Skipf("could not start sftp-server process: %v", err)
	}

	sftp, err := NewClientPipe(pr, pw)
	if err != nil {
		t.Fatal(err)
	}

	return sftp, cmd
}

func TestNewClient(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()

	if err := sftp.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestClientLstat(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	want, err := os.Lstat(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	got, err := sftp.Lstat(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	if !sameFile(want, got) {
		t.Fatalf("Lstat(%q): want %#v, got %#v", f.Name(), want, got)
	}
}

func TestClientLstatIsNotExist(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	os.Remove(f.Name())

	if _, err := sftp.Lstat(f.Name()); !os.IsNotExist(err) {
		t.Errorf("os.IsNotExist(%v) = false, want true", err)
	}
}

func TestClientMkdir(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	dir, err := ioutil.TempDir("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	sub := path.Join(dir, "mkdir1")
	if err := sftp.Mkdir(sub); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(sub); err != nil {
		t.Fatal(err)
	}
}

func TestClientOpen(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	got, err := sftp.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	if err := got.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestClientOpenIsNotExist(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	if _, err := sftp.Open("/doesnt/exist/"); !os.IsNotExist(err) {
		t.Errorf("os.IsNotExist(%v) = false, want true", err)
	}
}

func TestClientStatIsNotExist(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	if _, err := sftp.Stat("/doesnt/exist/"); !os.IsNotExist(err) {
		t.Errorf("os.IsNotExist(%v) = false, want true", err)
	}
}

const seekBytes = 128 * 1024

type seek struct {
	offset int64
}

func (s seek) Generate(r *rand.Rand, _ int) reflect.Value {
	s.offset = int64(r.Int31n(seekBytes))
	return reflect.ValueOf(s)
}

func (s seek) set(t *testing.T, r io.ReadSeeker) {
	if _, err := r.Seek(s.offset, os.SEEK_SET); err != nil {
		t.Fatalf("error while seeking with %+v: %v", s, err)
	}
}

func (s seek) current(t *testing.T, r io.ReadSeeker) {
	const mid = seekBytes / 2

	skip := s.offset / 2
	if s.offset > mid {
		skip = -skip
	}

	if _, err := r.Seek(mid, os.SEEK_SET); err != nil {
		t.Fatalf("error seeking to midpoint with %+v: %v", s, err)
	}
	if _, err := r.Seek(skip, os.SEEK_CUR); err != nil {
		t.Fatalf("error seeking from %d with %+v: %v", mid, s, err)
	}
}

func (s seek) end(t *testing.T, r io.ReadSeeker) {
	if _, err := r.Seek(-s.offset, os.SEEK_END); err != nil {
		t.Fatalf("error seeking from end with %+v: %v", s, err)
	}
}

func TestClientSeek(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	fOS, err := ioutil.TempFile("", "seek-test")
	if err != nil {
		t.Fatal(err)
	}
	defer fOS.Close()

	fSFTP, err := sftp.Open(fOS.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer fSFTP.Close()

	writeN(t, fOS, seekBytes)

	if err := quick.CheckEqual(
		func(s seek) (string, int64) { s.set(t, fOS); return readHash(t, fOS) },
		func(s seek) (string, int64) { s.set(t, fSFTP); return readHash(t, fSFTP) },
		nil,
	); err != nil {
		t.Errorf("Seek: expected equal absolute seeks: %v", err)
	}

	if err := quick.CheckEqual(
		func(s seek) (string, int64) { s.current(t, fOS); return readHash(t, fOS) },
		func(s seek) (string, int64) { s.current(t, fSFTP); return readHash(t, fSFTP) },
		nil,
	); err != nil {
		t.Errorf("Seek: expected equal seeks from middle: %v", err)
	}

	if err := quick.CheckEqual(
		func(s seek) (string, int64) { s.end(t, fOS); return readHash(t, fOS) },
		func(s seek) (string, int64) { s.end(t, fSFTP); return readHash(t, fSFTP) },
		nil,
	); err != nil {
		t.Errorf("Seek: expected equal seeks from end: %v", err)
	}
}

func TestClientCreate(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.Remove(f.Name())

	f2, err := sftp.Create(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
}

func TestClientAppend(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.Remove(f.Name())

	f2, err := sftp.OpenFile(f.Name(), os.O_RDWR|os.O_APPEND)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
}

func TestClientCreateFailed(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.Remove(f.Name())

	f2, err := sftp.Create(f.Name())
	if err1, ok := err.(*StatusError); !ok || err1.Code != ssh_FX_PERMISSION_DENIED {
		t.Fatalf("Create: want: %v, got %#v", ssh_FX_PERMISSION_DENIED, err)
	}
	if err == nil {
		f2.Close()
	}
}

func TestClientFileName(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	f2, err := sftp.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	if got, want := f2.Name(), f.Name(); got != want {
		t.Fatalf("Name: got %q want %q", want, got)
	}
}

func TestClientFileStat(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	want, err := os.Lstat(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	f2, err := sftp.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	got, err := f2.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if !sameFile(want, got) {
		t.Fatalf("Lstat(%q): want %#v, got %#v", f.Name(), want, got)
	}
}

func TestClientStatLink(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	realName := f.Name()
	linkName := f.Name() + ".softlink"

	// create a symlink that points at sftptest
	if err := os.Symlink(realName, linkName); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(linkName)

	// compare Lstat of links
	wantLstat, err := os.Lstat(linkName)
	if err != nil {
		t.Fatal(err)
	}
	wantStat, err := os.Stat(linkName)
	if err != nil {
		t.Fatal(err)
	}

	gotLstat, err := sftp.Lstat(linkName)
	if err != nil {
		t.Fatal(err)
	}
	gotStat, err := sftp.Stat(linkName)
	if err != nil {
		t.Fatal(err)
	}

	// check that stat is not lstat from os package
	if sameFile(wantLstat, wantStat) {
		t.Fatalf("Lstat / Stat(%q): both %#v %#v", f.Name(), wantLstat, wantStat)
	}

	// compare Lstat of links
	if !sameFile(wantLstat, gotLstat) {
		t.Fatalf("Lstat(%q): want %#v, got %#v", f.Name(), wantLstat, gotLstat)
	}

	// compare Stat of links
	if !sameFile(wantStat, gotStat) {
		t.Fatalf("Stat(%q): want %#v, got %#v", f.Name(), wantStat, gotStat)
	}

	// check that stat is not lstat
	if sameFile(gotLstat, gotStat) {
		t.Fatalf("Lstat / Stat(%q): both %#v %#v", f.Name(), gotLstat, gotStat)
	}
}

func TestClientRemove(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Remove(f.Name()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(f.Name()); !os.IsNotExist(err) {
		t.Fatal(err)
	}
}

func TestClientRemoveDir(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	dir, err := ioutil.TempDir("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Remove(dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(dir); !os.IsNotExist(err) {
		t.Fatal(err)
	}
}

func TestClientRemoveFailed(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Remove(f.Name()); err == nil {
		t.Fatalf("Remove(%v): want: permission denied, got %v", f.Name(), err)
	}
	if _, err := os.Lstat(f.Name()); err != nil {
		t.Fatal(err)
	}
}

func TestClientRename(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	f2 := f.Name() + ".new"
	if err := sftp.Rename(f.Name(), f2); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(f.Name()); !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if _, err := os.Lstat(f2); err != nil {
		t.Fatal(err)
	}
}

func TestClientGetwd(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	lwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	rwd, err := sftp.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if !path.IsAbs(rwd) {
		t.Fatalf("Getwd: wanted absolute path, got %q", rwd)
	}
	if lwd != rwd {
		t.Fatalf("Getwd: want %q, got %q", lwd, rwd)
	}
}

func TestClientReadLink(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	f2 := f.Name() + ".sym"
	if err := os.Symlink(f.Name(), f2); err != nil {
		t.Fatal(err)
	}
	if rl, err := sftp.ReadLink(f2); err != nil {
		t.Fatal(err)
	} else if rl != f.Name() {
		t.Fatalf("unexpected link target: %v, not %v", rl, f.Name())
	}
}

func TestClientSymlink(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	f2 := f.Name() + ".sym"
	if err := sftp.Symlink(f.Name(), f2); err != nil {
		t.Fatal(err)
	}
	if rl, err := sftp.ReadLink(f2); err != nil {
		t.Fatal(err)
	} else if rl != f.Name() {
		t.Fatalf("unexpected link target: %v, not %v", rl, f.Name())
	}
}

func TestClientChmod(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Chmod(f.Name(), 0531); err != nil {
		t.Fatal(err)
	}
	if stat, err := os.Stat(f.Name()); err != nil {
		t.Fatal(err)
	} else if stat.Mode()&os.ModePerm != 0531 {
		t.Fatalf("invalid perm %o\n", stat.Mode())
	}
}

func TestClientChmodReadonly(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Chmod(f.Name(), 0531); err == nil {
		t.Fatal("expected error")
	}
}

func TestClientChown(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	usr, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	chownto, err := user.Lookup("daemon") // seems common-ish...
	if err != nil {
		t.Fatal(err)
	}

	if usr.Uid != "0" {
		t.Log("must be root to run chown tests")
		t.Skip()
	}
	toUID, err := strconv.Atoi(chownto.Uid)
	if err != nil {
		t.Fatal(err)
	}
	toGID, err := strconv.Atoi(chownto.Gid)
	if err != nil {
		t.Fatal(err)
	}

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	before, err := exec.Command("ls", "-nl", f.Name()).Output()
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Chown(f.Name(), toUID, toGID); err != nil {
		t.Fatal(err)
	}
	after, err := exec.Command("ls", "-nl", f.Name()).Output()
	if err != nil {
		t.Fatal(err)
	}

	spaceRegex := regexp.MustCompile(`\s+`)

	beforeWords := spaceRegex.Split(string(before), -1)
	if beforeWords[2] != "0" {
		t.Fatalf("bad previous user? should be root")
	}
	afterWords := spaceRegex.Split(string(after), -1)
	if afterWords[2] != chownto.Uid || afterWords[3] != chownto.Gid {
		t.Fatalf("bad chown: %#v", afterWords)
	}
	t.Logf("before: %v", string(before))
	t.Logf(" after: %v", string(after))
}

func TestClientChownReadonly(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	usr, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	chownto, err := user.Lookup("daemon") // seems common-ish...
	if err != nil {
		t.Fatal(err)
	}

	if usr.Uid != "0" {
		t.Log("must be root to run chown tests")
		t.Skip()
	}
	toUID, err := strconv.Atoi(chownto.Uid)
	if err != nil {
		t.Fatal(err)
	}
	toGID, err := strconv.Atoi(chownto.Gid)
	if err != nil {
		t.Fatal(err)
	}

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	if err := sftp.Chown(f.Name(), toUID, toGID); err == nil {
		t.Fatal("expected error")
	}
}

func TestClientChtimes(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}

	atime := time.Date(2013, 2, 23, 13, 24, 35, 0, time.UTC)
	mtime := time.Date(1985, 6, 12, 6, 6, 6, 0, time.UTC)
	if err := sftp.Chtimes(f.Name(), atime, mtime); err != nil {
		t.Fatal(err)
	}
	if stat, err := os.Stat(f.Name()); err != nil {
		t.Fatal(err)
	} else if stat.ModTime().Sub(mtime) != 0 {
		t.Fatalf("incorrect mtime: %v vs %v", stat.ModTime(), mtime)
	}
}

func TestClientChtimesReadonly(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}

	atime := time.Date(2013, 2, 23, 13, 24, 35, 0, time.UTC)
	mtime := time.Date(1985, 6, 12, 6, 6, 6, 0, time.UTC)
	if err := sftp.Chtimes(f.Name(), atime, mtime); err == nil {
		t.Fatal("expected error")
	}
}

func TestClientTruncate(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	fname := f.Name()

	if n, err := f.Write([]byte("hello world")); n != 11 || err != nil {
		t.Fatal(err)
	}
	f.Close()

	if err := sftp.Truncate(fname, 5); err != nil {
		t.Fatal(err)
	}
	if stat, err := os.Stat(fname); err != nil {
		t.Fatal(err)
	} else if stat.Size() != 5 {
		t.Fatalf("unexpected size: %d", stat.Size())
	}
}

func TestClientTruncateReadonly(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	fname := f.Name()

	if n, err := f.Write([]byte("hello world")); n != 11 || err != nil {
		t.Fatal(err)
	}
	f.Close()

	if err := sftp.Truncate(fname, 5); err == nil {
		t.Fatal("expected error")
	}
	if stat, err := os.Stat(fname); err != nil {
		t.Fatal(err)
	} else if stat.Size() != 11 {
		t.Fatalf("unexpected size: %d", stat.Size())
	}
}

func sameFile(want, got os.FileInfo) bool {
	return want.Name() == got.Name() &&
		want.Size() == got.Size()
}

func TestClientReadSimple(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	d, err := ioutil.TempDir("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(d)

	f, err := ioutil.TempFile(d, "read-test")
	if err != nil {
		t.Fatal(err)
	}
	fname := f.Name()
	f.Write([]byte("hello"))
	f.Close()

	f2, err := sftp.Open(fname)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	stuff := make([]byte, 32)
	n, err := f2.Read(stuff)
	if err != nil && err != io.EOF {
		t.Fatalf("err: %v", err)
	}
	if n != 5 {
		t.Fatalf("n should be 5, is %v", n)
	}
	if string(stuff[0:5]) != "hello" {
		t.Fatalf("invalid contents")
	}
}

func TestClientReadDir(t *testing.T) {
	sftp1, cmd1 := testClient(t, READONLY, NO_DELAY)
	sftp2, cmd2 := testClientGoSvr(t, READONLY, NO_DELAY)
	defer cmd1.Wait()
	defer cmd2.Wait()
	defer sftp1.Close()
	defer sftp2.Close()

	dir := "/dev/"

	d, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	osfiles, err := d.Readdir(4096)
	if err != nil {
		t.Fatal(err)
	}

	sftp1Files, err := sftp1.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	sftp2Files, err := sftp2.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	osFilesByName := map[string]os.FileInfo{}
	for _, f := range osfiles {
		osFilesByName[f.Name()] = f
	}
	sftp1FilesByName := map[string]os.FileInfo{}
	for _, f := range sftp1Files {
		sftp1FilesByName[f.Name()] = f
	}
	sftp2FilesByName := map[string]os.FileInfo{}
	for _, f := range sftp2Files {
		sftp2FilesByName[f.Name()] = f
	}

	if len(osFilesByName) != len(sftp1FilesByName) || len(sftp1FilesByName) != len(sftp2FilesByName) {
		t.Fatalf("os gives %v, sftp1 gives %v, sftp2 gives %v", len(osFilesByName), len(sftp1FilesByName), len(sftp2FilesByName))
	}

	for name, osF := range osFilesByName {
		sftp1F, ok := sftp1FilesByName[name]
		if !ok {
			t.Fatalf("%v present in os but not sftp1", name)
		}
		sftp2F, ok := sftp2FilesByName[name]
		if !ok {
			t.Fatalf("%v present in os but not sftp2", name)
		}

		//t.Logf("%v: %v %v %v", name, osF, sftp1F, sftp2F)
		if osF.Size() != sftp1F.Size() || sftp1F.Size() != sftp2F.Size() {
			t.Fatalf("size %v %v %v", osF.Size(), sftp1F.Size(), sftp2F.Size())
		}
		if osF.IsDir() != sftp1F.IsDir() || sftp1F.IsDir() != sftp2F.IsDir() {
			t.Fatalf("isdir %v %v %v", osF.IsDir(), sftp1F.IsDir(), sftp2F.IsDir())
		}
		if osF.ModTime().Sub(sftp1F.ModTime()) > time.Second || sftp1F.ModTime() != sftp2F.ModTime() {
			t.Fatalf("modtime %v %v %v", osF.ModTime(), sftp1F.ModTime(), sftp2F.ModTime())
		}
		if osF.Mode() != sftp1F.Mode() || sftp1F.Mode() != sftp2F.Mode() {
			t.Fatalf("mode %x %x %x", osF.Mode(), sftp1F.Mode(), sftp2F.Mode())
		}
	}
}

var clientReadTests = []struct {
	n int64
}{
	{0},
	{1},
	{1000},
	{1024},
	{1025},
	{2048},
	{4096},
	{1 << 12},
	{1 << 13},
	{1 << 14},
	{1 << 15},
	{1 << 16},
	{1 << 17},
	{1 << 18},
	{1 << 19},
	{1 << 20},
}

func TestClientRead(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	d, err := ioutil.TempDir("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(d)

	for _, tt := range clientReadTests {
		f, err := ioutil.TempFile(d, "read-test")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		hash := writeN(t, f, tt.n)
		f2, err := sftp.Open(f.Name())
		if err != nil {
			t.Fatal(err)
		}
		defer f2.Close()
		hash2, n := readHash(t, f2)
		if hash != hash2 || tt.n != n {
			t.Errorf("Read: hash: want: %q, got %q, read: want: %v, got %v", hash, hash2, tt.n, n)
		}
	}
}

// readHash reads r until EOF returning the number of bytes read
// and the hash of the contents.
func readHash(t *testing.T, r io.Reader) (string, int64) {
	h := sha1.New()
	tr := io.TeeReader(r, h)
	read, err := io.Copy(ioutil.Discard, tr)
	if err != nil {
		t.Fatal(err)
	}
	return string(h.Sum(nil)), read
}

// writeN writes n bytes of random data to w and returns the
// hash of that data.
func writeN(t *testing.T, w io.Writer, n int64) string {
	rand, err := os.Open("/dev/urandom")
	if err != nil {
		t.Fatal(err)
	}
	defer rand.Close()

	h := sha1.New()

	mw := io.MultiWriter(w, h)

	written, err := io.CopyN(mw, rand, n)
	if err != nil {
		t.Fatal(err)
	}
	if written != n {
		t.Fatalf("CopyN(%v): wrote: %v", n, written)
	}
	return string(h.Sum(nil))
}

var clientWriteTests = []struct {
	n     int
	total int64 // cumulative file size
}{
	{0, 0},
	{1, 1},
	{0, 1},
	{999, 1000},
	{24, 1024},
	{1023, 2047},
	{2048, 4095},
	{1 << 12, 8191},
	{1 << 13, 16383},
	{1 << 14, 32767},
	{1 << 15, 65535},
	{1 << 16, 131071},
	{1 << 17, 262143},
	{1 << 18, 524287},
	{1 << 19, 1048575},
	{1 << 20, 2097151},
	{1 << 21, 4194303},
}

func TestClientWrite(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	d, err := ioutil.TempDir("", "sftptest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(d)

	f := path.Join(d, "writeTest")
	w, err := sftp.Create(f)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	for _, tt := range clientWriteTests {
		got, err := w.Write(make([]byte, tt.n))
		if err != nil {
			t.Fatal(err)
		}
		if got != tt.n {
			t.Errorf("Write(%v): wrote: want: %v, got %v", tt.n, tt.n, got)
		}
		fi, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		if total := fi.Size(); total != tt.total {
			t.Errorf("Write(%v): size: want: %v, got %v", tt.n, tt.total, total)
		}
	}
}

// taken from github.com/kr/fs/walk_test.go

type Node struct {
	name    string
	entries []*Node // nil if the entry is a file
	mark    int
}

var tree = &Node{
	"testdata",
	[]*Node{
		{"a", nil, 0},
		{"b", []*Node{}, 0},
		{"c", nil, 0},
		{
			"d",
			[]*Node{
				{"x", nil, 0},
				{"y", []*Node{}, 0},
				{
					"z",
					[]*Node{
						{"u", nil, 0},
						{"v", nil, 0},
					},
					0,
				},
			},
			0,
		},
	},
	0,
}

func walkTree(n *Node, path string, f func(path string, n *Node)) {
	f(path, n)
	for _, e := range n.entries {
		walkTree(e, filepath.Join(path, e.name), f)
	}
}

func makeTree(t *testing.T) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.entries == nil {
			fd, err := os.Create(path)
			if err != nil {
				t.Errorf("makeTree: %v", err)
				return
			}
			fd.Close()
		} else {
			os.Mkdir(path, 0770)
		}
	})
}

func markTree(n *Node) { walkTree(n, "", func(path string, n *Node) { n.mark++ }) }

func checkMarks(t *testing.T, report bool) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.mark != 1 && report {
			t.Errorf("node %s mark = %d; expected 1", path, n.mark)
		}
		n.mark = 0
	})
}

// Assumes that each node name is unique. Good enough for a test.
// If clear is true, any incoming error is cleared before return. The errors
// are always accumulated, though.
func mark(path string, info os.FileInfo, err error, errors *[]error, clear bool) error {
	if err != nil {
		*errors = append(*errors, err)
		if clear {
			return nil
		}
		return err
	}
	name := info.Name()
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++
		}
	})
	return nil
}

func TestClientWalk(t *testing.T) {
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	makeTree(t)
	errors := make([]error, 0, 10)
	clear := true
	markFn := func(walker *fs.Walker) error {
		for walker.Step() {
			err := mark(walker.Path(), walker.Stat(), walker.Err(), &errors, clear)
			if err != nil {
				return err
			}
		}
		return nil
	}
	// Expect no errors.
	err := markFn(sftp.Walk(tree.name))
	if err != nil {
		t.Fatalf("no error expected, found: %s", err)
	}
	if len(errors) != 0 {
		t.Fatalf("unexpected errors: %s", errors)
	}
	checkMarks(t, true)
	errors = errors[0:0]

	// Test permission errors.  Only possible if we're not root
	// and only on some file systems (AFS, FAT).  To avoid errors during
	// all.bash on those file systems, skip during go test -short.
	if os.Getuid() > 0 && !testing.Short() {
		// introduce 2 errors: chmod top-level directories to 0
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0)

		// 3) capture errors, expect two.
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark--
		tree.entries[3].mark--
		err := markFn(sftp.Walk(tree.name))
		if err != nil {
			t.Fatalf("expected no error return from Walk, got %s", err)
		}
		if len(errors) != 2 {
			t.Errorf("expected 2 errors, got %d: %s", len(errors), errors)
		}
		// the inaccessible subtrees were marked manually
		checkMarks(t, true)
		errors = errors[0:0]

		// 4) capture errors, stop after first error.
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark--
		tree.entries[3].mark--
		clear = false // error will stop processing
		err = markFn(sftp.Walk(tree.name))
		if err == nil {
			t.Fatalf("expected error return from Walk")
		}
		if len(errors) != 1 {
			t.Errorf("expected 1 error, got %d: %s", len(errors), errors)
		}
		// the inaccessible subtrees were marked manually
		checkMarks(t, false)
		errors = errors[0:0]

		// restore permissions
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0770)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0770)
	}

	// cleanup
	if err := os.RemoveAll(tree.name); err != nil {
		t.Errorf("removeTree: %v", err)
	}
}

// sftp/issue/42, abrupt server hangup would result in client hangs.
func TestServerRoughDisconnect(t *testing.T) {
	if *testServerImpl {
		t.Skipf("skipping with -testserver")
	}
	sftp, cmd := testClient(t, READONLY, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := sftp.Open("/dev/zero")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	go func() {
		time.Sleep(100 * time.Millisecond)
		cmd.Process.Kill()
	}()

	io.Copy(ioutil.Discard, f)
}

// sftp/issue/26 writing to a read only file caused client to loop.
func TestClientWriteToROFile(t *testing.T) {
	sftp, cmd := testClient(t, READWRITE, NO_DELAY)
	defer cmd.Wait()
	defer sftp.Close()

	f, err := sftp.Open("/dev/zero")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, err = f.Write([]byte("hello"))
	if err == nil {
		t.Fatal("expected error, got", err)
	}
}

func benchmarkRead(b *testing.B, bufsize int, delay time.Duration) {
	size := 10*1024*1024 + 123 // ~10MiB

	// open sftp client
	sftp, cmd := testClient(b, READONLY, delay)
	defer cmd.Wait()
	defer sftp.Close()

	buf := make([]byte, bufsize)

	b.ResetTimer()
	b.SetBytes(int64(size))

	for i := 0; i < b.N; i++ {
		offset := 0

		f2, err := sftp.Open("/dev/zero")
		if err != nil {
			b.Fatal(err)
		}
		defer f2.Close()

		for offset < size {
			n, err := io.ReadFull(f2, buf)
			offset += n
			if err == io.ErrUnexpectedEOF && offset != size {
				b.Fatalf("read too few bytes! want: %d, got: %d", size, n)
			}

			if err != nil {
				b.Fatal(err)
			}

			offset += n
		}
	}
}

func BenchmarkRead1k(b *testing.B) {
	benchmarkRead(b, 1*1024, NO_DELAY)
}

func BenchmarkRead16k(b *testing.B) {
	benchmarkRead(b, 16*1024, NO_DELAY)
}

func BenchmarkRead32k(b *testing.B) {
	benchmarkRead(b, 32*1024, NO_DELAY)
}

func BenchmarkRead128k(b *testing.B) {
	benchmarkRead(b, 128*1024, NO_DELAY)
}

func BenchmarkRead512k(b *testing.B) {
	benchmarkRead(b, 512*1024, NO_DELAY)
}

func BenchmarkRead1MiB(b *testing.B) {
	benchmarkRead(b, 1024*1024, NO_DELAY)
}

func BenchmarkRead4MiB(b *testing.B) {
	benchmarkRead(b, 4*1024*1024, NO_DELAY)
}

func BenchmarkRead4MiBDelay10Msec(b *testing.B) {
	benchmarkRead(b, 4*1024*1024, 10*time.Millisecond)
}

func BenchmarkRead4MiBDelay50Msec(b *testing.B) {
	benchmarkRead(b, 4*1024*1024, 50*time.Millisecond)
}

func BenchmarkRead4MiBDelay150Msec(b *testing.B) {
	benchmarkRead(b, 4*1024*1024, 150*time.Millisecond)
}

func benchmarkWrite(b *testing.B, bufsize int, delay time.Duration) {
	size := 10*1024*1024 + 123 // ~10MiB

	// open sftp client
	sftp, cmd := testClient(b, false, delay)
	defer cmd.Wait()
	defer sftp.Close()

	data := make([]byte, size)

	b.ResetTimer()
	b.SetBytes(int64(size))

	for i := 0; i < b.N; i++ {
		offset := 0

		f, err := ioutil.TempFile("", "sftptest")
		if err != nil {
			b.Fatal(err)
		}
		defer os.Remove(f.Name())

		f2, err := sftp.Create(f.Name())
		if err != nil {
			b.Fatal(err)
		}
		defer f2.Close()

		for offset < size {
			n, err := f2.Write(data[offset:min(len(data), offset+bufsize)])
			if err != nil {
				b.Fatal(err)
			}

			if offset+n < size && n != bufsize {
				b.Fatalf("wrote too few bytes! want: %d, got: %d", size, n)
			}

			offset += n
		}

		f2.Close()

		fi, err := os.Stat(f.Name())
		if err != nil {
			b.Fatal(err)
		}

		if fi.Size() != int64(size) {
			b.Fatalf("wrong file size: want %d, got %d", size, fi.Size())
		}

		os.Remove(f.Name())
	}
}

func BenchmarkWrite1k(b *testing.B) {
	benchmarkWrite(b, 1*1024, NO_DELAY)
}

func BenchmarkWrite16k(b *testing.B) {
	benchmarkWrite(b, 16*1024, NO_DELAY)
}

func BenchmarkWrite32k(b *testing.B) {
	benchmarkWrite(b, 32*1024, NO_DELAY)
}

func BenchmarkWrite128k(b *testing.B) {
	benchmarkWrite(b, 128*1024, NO_DELAY)
}

func BenchmarkWrite512k(b *testing.B) {
	benchmarkWrite(b, 512*1024, NO_DELAY)
}

func BenchmarkWrite1MiB(b *testing.B) {
	benchmarkWrite(b, 1024*1024, NO_DELAY)
}

func BenchmarkWrite4MiB(b *testing.B) {
	benchmarkWrite(b, 4*1024*1024, NO_DELAY)
}

func BenchmarkWrite4MiBDelay10Msec(b *testing.B) {
	benchmarkWrite(b, 4*1024*1024, 10*time.Millisecond)
}

func BenchmarkWrite4MiBDelay50Msec(b *testing.B) {
	benchmarkWrite(b, 4*1024*1024, 50*time.Millisecond)
}

func BenchmarkWrite4MiBDelay150Msec(b *testing.B) {
	benchmarkWrite(b, 4*1024*1024, 150*time.Millisecond)
}

func benchmarkCopyDown(b *testing.B, fileSize int64, delay time.Duration) {
	// Create a temp file and fill it with zero's.
	src, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		b.Fatal(err)
	}
	defer src.Close()
	srcFilename := src.Name()
	defer os.Remove(srcFilename)
	zero, err := os.Open("/dev/zero")
	if err != nil {
		b.Fatal(err)
	}
	n, err := io.Copy(src, io.LimitReader(zero, fileSize))
	if err != nil {
		b.Fatal(err)
	}
	if n < fileSize {
		b.Fatal("short copy")
	}
	zero.Close()
	src.Close()

	sftp, cmd := testClient(b, READONLY, delay)
	defer cmd.Wait()
	defer sftp.Close()
	b.ResetTimer()
	b.SetBytes(fileSize)

	for i := 0; i < b.N; i++ {
		dst, err := ioutil.TempFile("", "sftptest")
		if err != nil {
			b.Fatal(err)
		}
		defer os.Remove(dst.Name())

		src, err := sftp.Open(srcFilename)
		if err != nil {
			b.Fatal(err)
		}
		defer src.Close()
		n, err := io.Copy(dst, src)
		if err != nil {
			b.Fatalf("copy error: %v", err)
		}
		if n < fileSize {
			b.Fatal("unable to copy all bytes")
		}
		dst.Close()
		fi, err := os.Stat(dst.Name())
		if err != nil {
			b.Fatal(err)
		}

		if fi.Size() != fileSize {
			b.Fatalf("wrong file size: want %d, got %d", fileSize, fi.Size())
		}
		os.Remove(dst.Name())
	}
}

func BenchmarkCopyDown10MiBDelay10Msec(b *testing.B) {
	benchmarkCopyDown(b, 10*1024*1024, 10*time.Millisecond)
}

func BenchmarkCopyDown10MiBDelay50Msec(b *testing.B) {
	benchmarkCopyDown(b, 10*1024*1024, 50*time.Millisecond)
}

func BenchmarkCopyDown10MiBDelay150Msec(b *testing.B) {
	benchmarkCopyDown(b, 10*1024*1024, 150*time.Millisecond)
}

func benchmarkCopyUp(b *testing.B, fileSize int64, delay time.Duration) {
	// Create a temp file and fill it with zero's.
	src, err := ioutil.TempFile("", "sftptest")
	if err != nil {
		b.Fatal(err)
	}
	defer src.Close()
	srcFilename := src.Name()
	defer os.Remove(srcFilename)
	zero, err := os.Open("/dev/zero")
	if err != nil {
		b.Fatal(err)
	}
	n, err := io.Copy(src, io.LimitReader(zero, fileSize))
	if err != nil {
		b.Fatal(err)
	}
	if n < fileSize {
		b.Fatal("short copy")
	}
	zero.Close()
	src.Close()

	sftp, cmd := testClient(b, false, delay)
	defer cmd.Wait()
	defer sftp.Close()

	b.ResetTimer()
	b.SetBytes(fileSize)

	for i := 0; i < b.N; i++ {
		tmp, err := ioutil.TempFile("", "sftptest")
		if err != nil {
			b.Fatal(err)
		}
		tmp.Close()
		defer os.Remove(tmp.Name())

		dst, err := sftp.Create(tmp.Name())
		if err != nil {
			b.Fatal(err)
		}
		defer dst.Close()
		src, err := os.Open(srcFilename)
		if err != nil {
			b.Fatal(err)
		}
		defer src.Close()
		n, err := io.Copy(dst, src)
		if err != nil {
			b.Fatalf("copy error: %v", err)
		}
		if n < fileSize {
			b.Fatal("unable to copy all bytes")
		}

		fi, err := os.Stat(tmp.Name())
		if err != nil {
			b.Fatal(err)
		}

		if fi.Size() != fileSize {
			b.Fatalf("wrong file size: want %d, got %d", fileSize, fi.Size())
		}
		os.Remove(tmp.Name())
	}
}

func BenchmarkCopyUp10MiBDelay10Msec(b *testing.B) {
	benchmarkCopyUp(b, 10*1024*1024, 10*time.Millisecond)
}

func BenchmarkCopyUp10MiBDelay50Msec(b *testing.B) {
	benchmarkCopyUp(b, 10*1024*1024, 50*time.Millisecond)
}

func BenchmarkCopyUp10MiBDelay150Msec(b *testing.B) {
	benchmarkCopyUp(b, 10*1024*1024, 150*time.Millisecond)
}
