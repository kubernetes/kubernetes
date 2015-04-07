/*
Package gbytes provides a buffer that supports incrementally detecting input.

You use gbytes.Buffer with the gbytes.Say matcher.  When Say finds a match, it fastforwards the buffer's read cursor to the end of that match.

Subsequent matches against the buffer will only operate against data that appears *after* the read cursor.

The read cursor is an opaque implementation detail that you cannot access.  You should use the Say matcher to sift through the buffer.  You can always
access the entire buffer's contents with Contents().

*/
package gbytes

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"sync"
	"time"
)

/*
gbytes.Buffer implements an io.Writer and can be used with the gbytes.Say matcher.

You should only use a gbytes.Buffer in test code.  It stores all writes in an in-memory buffer - behavior that is inappropriate for production code!
*/
type Buffer struct {
	contents     []byte
	readCursor   uint64
	lock         *sync.Mutex
	detectCloser chan interface{}
	closed       bool
}

/*
NewBuffer returns a new gbytes.Buffer
*/
func NewBuffer() *Buffer {
	return &Buffer{
		lock: &sync.Mutex{},
	}
}

/*
BufferWithBytes returns a new gbytes.Buffer seeded with the passed in bytes
*/
func BufferWithBytes(bytes []byte) *Buffer {
	return &Buffer{
		lock:     &sync.Mutex{},
		contents: bytes,
	}
}

/*
Write implements the io.Writer interface
*/
func (b *Buffer) Write(p []byte) (n int, err error) {
	b.lock.Lock()
	defer b.lock.Unlock()

	if b.closed {
		return 0, errors.New("attempt to write to closed buffer")
	}

	b.contents = append(b.contents, p...)
	return len(p), nil
}

/*
Read implements the io.Reader interface. It advances the
cursor as it reads.

Returns an error if called after Close.
*/
func (b *Buffer) Read(d []byte) (int, error) {
	b.lock.Lock()
	defer b.lock.Unlock()

	if b.closed {
		return 0, errors.New("attempt to read from closed buffer")
	}

	if uint64(len(b.contents)) <= b.readCursor {
		return 0, io.EOF
	}

	n := copy(d, b.contents[b.readCursor:])
	b.readCursor += uint64(n)

	return n, nil
}

/*
Close signifies that the buffer will no longer be written to
*/
func (b *Buffer) Close() error {
	b.lock.Lock()
	defer b.lock.Unlock()

	b.closed = true

	return nil
}

/*
Closed returns true if the buffer has been closed
*/
func (b *Buffer) Closed() bool {
	b.lock.Lock()
	defer b.lock.Unlock()

	return b.closed
}

/*
Contents returns all data ever written to the buffer.
*/
func (b *Buffer) Contents() []byte {
	b.lock.Lock()
	defer b.lock.Unlock()

	contents := make([]byte, len(b.contents))
	copy(contents, b.contents)
	return contents
}

/*
Detect takes a regular expression and returns a channel.

The channel will receive true the first time data matching the regular expression is written to the buffer.
The channel is subsequently closed and the buffer's read-cursor is fast-forwarded to just after the matching region.

You typically don't need to use Detect and should use the ghttp.Say matcher instead.  Detect is useful, however, in cases where your code must
be branch and handle different outputs written to the buffer.

For example, consider a buffer hooked up to the stdout of a client library.  You may (or may not, depending on state outside of your control) need to authenticate the client library.

You could do something like:

select {
case <-buffer.Detect("You are not logged in"):
	//log in
case <-buffer.Detect("Success"):
	//carry on
case <-time.After(time.Second):
	//welp
}
buffer.CancelDetects()

You should always call CancelDetects after using Detect.  This will close any channels that have not detected and clean up the goroutines that were spawned to support them.

Finally, you can pass detect a format string followed by variadic arguments.  This will construct the regexp using fmt.Sprintf.
*/
func (b *Buffer) Detect(desired string, args ...interface{}) chan bool {
	formattedRegexp := desired
	if len(args) > 0 {
		formattedRegexp = fmt.Sprintf(desired, args...)
	}
	re := regexp.MustCompile(formattedRegexp)

	b.lock.Lock()
	defer b.lock.Unlock()

	if b.detectCloser == nil {
		b.detectCloser = make(chan interface{})
	}

	closer := b.detectCloser
	response := make(chan bool)
	go func() {
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		defer close(response)
		for {
			select {
			case <-ticker.C:
				b.lock.Lock()
				data, cursor := b.contents[b.readCursor:], b.readCursor
				loc := re.FindIndex(data)
				b.lock.Unlock()

				if loc != nil {
					response <- true
					b.lock.Lock()
					newCursorPosition := cursor + uint64(loc[1])
					if newCursorPosition >= b.readCursor {
						b.readCursor = newCursorPosition
					}
					b.lock.Unlock()
					return
				}
			case <-closer:
				return
			}
		}
	}()

	return response
}

/*
CancelDetects cancels any pending detects and cleans up their goroutines.  You should always call this when you're done with a set of Detect channels.
*/
func (b *Buffer) CancelDetects() {
	b.lock.Lock()
	defer b.lock.Unlock()

	close(b.detectCloser)
	b.detectCloser = nil
}

func (b *Buffer) didSay(re *regexp.Regexp) (bool, []byte) {
	b.lock.Lock()
	defer b.lock.Unlock()

	unreadBytes := b.contents[b.readCursor:]
	copyOfUnreadBytes := make([]byte, len(unreadBytes))
	copy(copyOfUnreadBytes, unreadBytes)

	loc := re.FindIndex(unreadBytes)

	if loc != nil {
		b.readCursor += uint64(loc[1])
		return true, copyOfUnreadBytes
	} else {
		return false, copyOfUnreadBytes
	}
}
