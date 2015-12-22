package meta

import (
	"io"
	"net"
)

// proxy brokers a connection from src to dst
func proxy(dst, src *net.TCPConn) error {
	// channels to wait on the close event for each connection
	serverClosed := make(chan struct{}, 1)
	clientClosed := make(chan struct{}, 1)
	errors := make(chan error, 2)

	go broker(dst, src, clientClosed, errors)
	go broker(src, dst, serverClosed, errors)

	// wait for one half of the proxy to exit, then trigger a shutdown of the
	// other half by calling CloseRead(). This will break the read loop in the
	// broker and allow us to fully close the connection cleanly without a
	// "use of closed network connection" error.
	var waitFor chan struct{}
	select {
	case <-clientClosed:
		// the client closed first and any more packets from the server aren't
		// useful, so we can optionally SetLinger(0) here to recycle the port
		// faster.
		dst.SetLinger(0)
		dst.Close()
		waitFor = serverClosed
	case <-serverClosed:
		src.Close()
		waitFor = clientClosed
	case err := <-errors:
		src.Close()
		dst.SetLinger(0)
		dst.Close()
		return err
	}

	// Wait for the other connection to close.
	<-waitFor
	return nil
}

// This does the actual data transfer.
// The broker only closes the Read side.
func broker(dst, src net.Conn, srcClosed chan struct{}, errors chan error) {
	// We can handle errors in a finer-grained manner by inlining io.Copy (it's
	// simple, and we drop the ReaderFrom or WriterTo checks for
	// net.Conn->net.Conn transfers, which aren't needed). This would also let
	// us adjust buffersize.
	_, err := io.Copy(dst, src)

	if err != nil {
		errors <- err
	}
	if err := src.Close(); err != nil {
		errors <- err
	}
	srcClosed <- struct{}{}
}
