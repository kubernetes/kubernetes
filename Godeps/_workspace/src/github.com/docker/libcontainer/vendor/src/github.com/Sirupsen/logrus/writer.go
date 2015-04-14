package logrus

import (
	"bufio"
	"io"
	"runtime"
)

func (logger *Logger) Writer() (*io.PipeWriter) {
	reader, writer := io.Pipe()

	go logger.writerScanner(reader)
	runtime.SetFinalizer(writer, writerFinalizer)

	return writer
}

func (logger *Logger) writerScanner(reader *io.PipeReader) {
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		logger.Print(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		logger.Errorf("Error while reading from Writer: %s", err)
	}
	reader.Close()
}

func writerFinalizer(writer *io.PipeWriter) {
	writer.Close()
}
