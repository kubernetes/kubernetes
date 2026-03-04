package logrus

import (
	"bufio"
	"io"
	"runtime"
	"strings"
)

// Writer at INFO level. See WriterLevel for details.
func (logger *Logger) Writer() *io.PipeWriter {
	return logger.WriterLevel(InfoLevel)
}

// WriterLevel returns an io.Writer that can be used to write arbitrary text to
// the logger at the given log level. Each line written to the writer will be
// printed in the usual way using formatters and hooks. The writer is part of an
// io.Pipe and it is the callers responsibility to close the writer when done.
// This can be used to override the standard library logger easily.
func (logger *Logger) WriterLevel(level Level) *io.PipeWriter {
	return NewEntry(logger).WriterLevel(level)
}

// Writer returns an io.Writer that writes to the logger at the info log level
func (entry *Entry) Writer() *io.PipeWriter {
	return entry.WriterLevel(InfoLevel)
}

// WriterLevel returns an io.Writer that writes to the logger at the given log level
func (entry *Entry) WriterLevel(level Level) *io.PipeWriter {
	reader, writer := io.Pipe()

	var printFunc func(args ...interface{})

	// Determine which log function to use based on the specified log level
	switch level {
	case TraceLevel:
		printFunc = entry.Trace
	case DebugLevel:
		printFunc = entry.Debug
	case InfoLevel:
		printFunc = entry.Info
	case WarnLevel:
		printFunc = entry.Warn
	case ErrorLevel:
		printFunc = entry.Error
	case FatalLevel:
		printFunc = entry.Fatal
	case PanicLevel:
		printFunc = entry.Panic
	default:
		printFunc = entry.Print
	}

	// Start a new goroutine to scan the input and write it to the logger using the specified print function.
	// It splits the input into chunks of up to 64KB to avoid buffer overflows.
	go entry.writerScanner(reader, printFunc)

	// Set a finalizer function to close the writer when it is garbage collected
	runtime.SetFinalizer(writer, writerFinalizer)

	return writer
}

// writerScanner scans the input from the reader and writes it to the logger
func (entry *Entry) writerScanner(reader *io.PipeReader, printFunc func(args ...interface{})) {
	scanner := bufio.NewScanner(reader)

	// Set the buffer size to the maximum token size to avoid buffer overflows
	scanner.Buffer(make([]byte, bufio.MaxScanTokenSize), bufio.MaxScanTokenSize)

	// Define a split function to split the input into chunks of up to 64KB
	chunkSize := bufio.MaxScanTokenSize // 64KB
	splitFunc := func(data []byte, atEOF bool) (int, []byte, error) {
		if len(data) >= chunkSize {
			return chunkSize, data[:chunkSize], nil
		}

		return bufio.ScanLines(data, atEOF)
	}

	// Use the custom split function to split the input
	scanner.Split(splitFunc)

	// Scan the input and write it to the logger using the specified print function
	for scanner.Scan() {
		printFunc(strings.TrimRight(scanner.Text(), "\r\n"))
	}

	// If there was an error while scanning the input, log an error
	if err := scanner.Err(); err != nil {
		entry.Errorf("Error while reading from Writer: %s", err)
	}

	// Close the reader when we are done
	reader.Close()
}

// WriterFinalizer is a finalizer function that closes then given writer when it is garbage collected
func writerFinalizer(writer *io.PipeWriter) {
	writer.Close()
}
