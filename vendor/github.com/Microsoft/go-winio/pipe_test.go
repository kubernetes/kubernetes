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

var aLongTimeAgo = time.Unix(1, 0)

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

func TestTimeoutPendingRead(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	serverDone := make(chan struct{})

	go func() {
		s, err := l.Accept()
		if err != nil {
			t.Fatal(err)
		}
		time.Sleep(1 * time.Second)
		s.Close()
		close(serverDone)
	}()

	client, err := DialPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	clientErr := make(chan error)
	go func() {
		buf := make([]byte, 10)
		_, err = client.Read(buf)
		clientErr <- err
	}()

	time.Sleep(100 * time.Millisecond) // make *sure* the pipe is reading before we set the deadline
	client.SetReadDeadline(aLongTimeAgo)

	select {
	case err = <-clientErr:
		if err != ErrTimeout {
			t.Fatalf("expected ErrTimeout, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("timed out while waiting for read to cancel")
		<-clientErr
	}
	<-serverDone
}

func TestTimeoutPendingWrite(t *testing.T) {
	l, err := ListenPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	serverDone := make(chan struct{})

	go func() {
		s, err := l.Accept()
		if err != nil {
			t.Fatal(err)
		}
		time.Sleep(1 * time.Second)
		s.Close()
		close(serverDone)
	}()

	client, err := DialPipe(testPipeName, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	clientErr := make(chan error)
	go func() {
		_, err = client.Write([]byte("this should timeout"))
		clientErr <- err
	}()

	time.Sleep(100 * time.Millisecond) // make *sure* the pipe is writing before we set the deadline
	client.SetWriteDeadline(aLongTimeAgo)

	select {
	case err = <-clientErr:
		if err != ErrTimeout {
			t.Fatalf("expected ErrTimeout, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatalf("timed out while waiting for write to cancel")
		<-clientErr
	}
	<-serverDone
}

type CloseWriter interface {
	CloseWrite() error
}

func TestEchoWithMessaging(t *testing.T) {
	c := PipeConfig{
		MessageMode:      true,  // Use message mode so that CloseWrite() is supported
		InputBufferSize:  65536, // Use 64KB buffers to improve performance
		OutputBufferSize: 65536,
	}
	l, err := ListenPipe(testPipeName, &c)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	listenerDone := make(chan bool)
	clientDone := make(chan bool)
	go func() {
		// server echo
		conn, e := l.Accept()
		if e != nil {
			t.Fatal(e)
		}
		defer conn.Close()

		time.Sleep(500 * time.Millisecond) // make *sure* we don't begin to read before eof signal is sent
		io.Copy(conn, conn)
		conn.(CloseWriter).CloseWrite()
		close(listenerDone)
	}()
	timeout := 1 * time.Second
	client, err := DialPipe(testPipeName, &timeout)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	go func() {
		// client read back
		bytes := make([]byte, 2)
		n, e := client.Read(bytes)
		if e != nil {
			t.Fatal(e)
		}
		if n != 2 {
			t.Fatalf("expected 2 bytes, got %v", n)
		}
		close(clientDone)
	}()

	payload := make([]byte, 2)
	payload[0] = 0
	payload[1] = 1

	n, err := client.Write(payload)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("expected 2 bytes, got %v", n)
	}
	client.(CloseWriter).CloseWrite()
	<-listenerDone
	<-clientDone
}
