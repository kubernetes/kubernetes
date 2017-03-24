### fifo

Go package for handling fifos in a sane way.

```
// OpenFifo opens a fifo. Returns io.ReadWriteCloser.
// Context can be used to cancel this function until open(2) has not returned.
// Accepted flags:
// - syscall.O_CREAT - create new fifo if one doesn't exist
// - syscall.O_RDONLY - open fifo only from reader side
// - syscall.O_WRONLY - open fifo only from writer side
// - syscall.O_RDWR - open fifo from both sides, never block on syscall level
// - syscall.O_NONBLOCK - return io.ReadWriteCloser even if other side of the
//     fifo isn't open. read/write will be connected after the actual fifo is
//     open or after fifo is closed.
func OpenFifo(ctx context.Context, fn string, flag int, perm os.FileMode) (io.ReadWriteCloser, error)


// Read from a fifo to a byte array.
func (f *fifo) Read(b []byte) (int, error)


// Write from byte array to a fifo.
func (f *fifo) Write(b []byte) (int, error)


// Close the fifo. Next reads/writes will error. This method can also be used
// before open(2) has returned and fifo was never opened.
func (f *fifo) Close() error 
```