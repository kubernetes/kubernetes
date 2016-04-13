package winio

import (
	"bufio"
	"io"
	"net"
	"os"
	"syscall"
	"testing"
	"time"
)

var testPipeName = `\\.\pipe\winiotestpipe`

func TestDialUnknownFailsImmediately(t *testing.T) {
	_, err := DialPipe(testPipeName, nil)
	if err.(*os.PathError).Err != syscall.ENOENT {
		t.Fatalf("expected ENOENT got %v", err)
	}
}

func TestDialListenerTimesOut(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	var d = time.Duration(10 * time.Millisecond)
	_, err = DialPipe(testPipeName, &d)
	if err != ErrTimeout {
		t.Fatalf("expected ErrTimeout, got %v", err)
	}
}

func TestDialAccessDeniedWithRestrictedSD(t *testing.T) {
	c := PipeConfig{
		SecurityDescriptor: "D:P(A;;0x1200FF;;;WD)",
	}
	l, err := ListenPipe(testPipeName, &c)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	_, err = DialPipe(testPipeName, nil)
	if err.(*os.PathError).Err != syscall.ERROR_ACCESS_DENIED {
		t.Fatalf("expected ERROR_ACCESS_DENIED, got %v", err)
	}
}

func getConnection(cfg *PipeConfig) (client net.Conn, server net.Conn, err error) {
	l, err := ListenPipe(testPipeName, cfg)
	if err != nil {
		return
	}
	defer l.Close()

	type response struct {
		c   net.Conn
		err error
	}
	ch := make(chan response)
	go func() {
		c, err := l.Accept()
		ch <- response{c, err}
	}()

	c, err := DialPipe(testPipeName, nil)
	if err != nil {
		return
	}

	r := <-ch
	if err = r.err; err != nil {
		c.Close()
		return
	}

	client = c
	server = r.c
	return
}

func TestReadTimeout(t *testing.T) {
	c, s, err := getConnection(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	defer s.Close()

	c.SetReadDeadline(time.Now().Add(10 * time.Millisecond))

	buf := make([]byte, 10)
	_, err = c.Read(buf)
	if err != ErrTimeout {
		t.Fatalf("expected ErrTimeout, got %v", err)
	}
}

func server(l net.Listener, ch chan int) {
	c, err := l.Accept()
	if err != nil {
		panic(err)
	}
	rw := bufio.NewReadWriter(bufio.NewReader(c), bufio.NewWriter(c))
	s, err := rw.ReadString('\n')
	if err != nil {
		panic(err)
	}
	_, err = rw.WriteString("got " + s)
	if err != nil {
		panic(err)
	}
	err = rw.Flush()
	if err != nil {
		panic(err)
	}
	c.Close()
	ch <- 1
}

func TestFullListenDialReadWrite(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	ch := make(chan int)
	go server(l, ch)

	c, err := DialPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	rw := bufio.NewReadWriter(bufio.NewReader(c), bufio.NewWriter(c))
	_, err = rw.WriteString("hello world\n")
	if err != nil {
		t.Fatal(err)
	}
	err = rw.Flush()
	if err != nil {
		t.Fatal(err)
	}

	s, err := rw.ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	ms := "got hello world\n"
	if s != ms {
		t.Errorf("expected '%s', got '%s'", ms, s)
	}

	<-ch
}

func TestCloseAbortsListen(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}

	ch := make(chan error)
	go func() {
		_, err := l.Accept()
		ch <- err
	}()

	time.Sleep(30 * time.Millisecond)
	l.Close()

	err = <-ch
	if err != ErrPipeListenerClosed {
		t.Fatalf("expected ErrPipeListenerClosed, got %v", err)
	}
}

func ensureEOFOnClose(t *testing.T, r io.Reader, w io.Closer) {
	b := make([]byte, 10)
	w.Close()
	n, err := r.Read(b)
	if n > 0 {
		t.Errorf("unexpected byte count %d", n)
	}
	if err != io.EOF {
		t.Errorf("expected EOF: %v", err)
	}
}

func TestCloseClientEOFServer(t *testing.T) {
	c, s, err := getConnection(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	defer s.Close()
	ensureEOFOnClose(t, c, s)
}

func TestCloseServerEOFClient(t *testing.T) {
	c, s, err := getConnection(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	defer s.Close()
	ensureEOFOnClose(t, s, c)
}

func TestCloseWriteEOF(t *testing.T) {
	cfg := &PipeConfig{
		MessageMode: true,
	}
	c, s, err := getConnection(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	defer s.Close()

	type closeWriter interface {
		CloseWrite() error
	}

	err = c.(closeWriter).CloseWrite()
	if err != nil {
		t.Fatal(err)
	}

	b := make([]byte, 10)
	_, err = s.Read(b)
	if err != io.EOF {
		t.Fatal(err)
	}
}

func TestAcceptAfterCloseFails(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	l.Close()
	_, err = l.Accept()
	if err != ErrPipeListenerClosed {
		t.Fatalf("expected ErrPipeListenerClosed, got %v", err)
	}
}

func TestDialTimesOutByDefault(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	_, err = DialPipe(testPipeName, nil)
	if err != ErrTimeout {
		t.Fatalf("expected ErrTimeout, got %v", err)
	}
}
