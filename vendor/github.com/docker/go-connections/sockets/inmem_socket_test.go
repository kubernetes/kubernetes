package sockets

import "testing"

func TestInmemSocket(t *testing.T) {
	l := NewInmemSocket("test", 0)
	defer l.Close()
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			conn.Write([]byte("hello"))
			conn.Close()
		}
	}()

	conn, err := l.Dial("test", "test")
	if err != nil {
		t.Fatal(err)
	}

	buf := make([]byte, 5)
	_, err = conn.Read(buf)
	if err != nil {
		t.Fatal(err)
	}

	if string(buf) != "hello" {
		t.Fatalf("expected `hello`, got %s", string(buf))
	}

	l.Close()
	conn, err = l.Dial("test", "test")
	if err != errClosed {
		t.Fatalf("expected `errClosed` error, got %v", err)
	}
}
