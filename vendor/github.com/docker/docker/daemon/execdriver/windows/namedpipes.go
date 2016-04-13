// +build windows

package windows

import (
	"io"

	"github.com/Sirupsen/logrus"
	"github.com/natefinch/npipe"
)

// stdinAccept runs as a go function. It waits for the container system
// to accept our offer of a named pipe for stdin. Once accepted, if we are
// running "attached" to the container (eg docker run -i), then we spin up
// a further thread to copy anything from the client into the container.
//
// Important design note. This function is run as a go function for a very
// good reason. The named pipe Accept call is blocking until one of two things
// happen. Either someone connects to it, or it is forcibly closed. Let's
// assume that no-one connects to it, the only way otherwise the Run()
// method would continue is by closing it. However, as that would be the same
// thread, it can't close it. Hence we run as another thread allowing Run()
// to close the named pipe.
func stdinAccept(inListen *npipe.PipeListener, pipeName string, copyfrom io.ReadCloser) {

	// Wait for the pipe to be connected to by the shim
	logrus.Debugln("stdinAccept: Waiting on ", pipeName)
	stdinConn, err := inListen.Accept()
	if err != nil {
		logrus.Errorf("Failed to accept on pipe %s %s", pipeName, err)
		return
	}
	logrus.Debugln("Connected to ", stdinConn.RemoteAddr())

	// Anything that comes from the client stdin should be copied
	// across to the stdin named pipe of the container.
	if copyfrom != nil {
		go func() {
			defer stdinConn.Close()
			logrus.Debugln("Calling io.Copy on stdin")
			bytes, err := io.Copy(stdinConn, copyfrom)
			logrus.Debugf("Finished io.Copy on stdin bytes=%d err=%s pipe=%s", bytes, err, stdinConn.RemoteAddr())
		}()
	} else {
		defer stdinConn.Close()
	}
}

// stdouterrAccept runs as a go function. It waits for the container system to
// accept our offer of a named pipe - in fact two of them - one for stdout
// and one for stderr (we are called twice). Once the named pipe is accepted,
// if we are running "attached" to the container (eg docker run -i), then we
// spin up a further thread to copy anything from the containers output channels
// to the client.
func stdouterrAccept(outerrListen *npipe.PipeListener, pipeName string, copyto io.Writer) {

	// Wait for the pipe to be connected to by the shim
	logrus.Debugln("out/err: Waiting on ", pipeName)
	outerrConn, err := outerrListen.Accept()
	if err != nil {
		logrus.Errorf("Failed to accept on pipe %s %s", pipeName, err)
		return
	}
	logrus.Debugln("Connected to ", outerrConn.RemoteAddr())

	// Anything that comes from the container named pipe stdout/err should be copied
	// across to the stdout/err of the client
	if copyto != nil {
		go func() {
			defer outerrConn.Close()
			logrus.Debugln("Calling io.Copy on ", pipeName)
			bytes, err := io.Copy(copyto, outerrConn)
			logrus.Debugf("Copied %d bytes from pipe=%s", bytes, outerrConn.RemoteAddr())
			if err != nil {
				// Not fatal, just debug log it
				logrus.Debugf("Error hit during copy %s", err)
			}
		}()
	} else {
		defer outerrConn.Close()
	}
}
