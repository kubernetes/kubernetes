/*
Copyright 2016 Euan Kemp

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package kmsgparser implements a parser for the Linux `/dev/kmsg` format.
// More information about this format may be found here:
// https://www.kernel.org/doc/Documentation/ABI/testing/dev-kmsg
// Some parts of it are slightly inspired by rsyslog's contrib module:
// https://github.com/rsyslog/rsyslog/blob/v8.22.0/contrib/imkmsg/kmsg.c
//
// This package is forked from github.com/euank/go-kmsg-parser/v3 at v3.0.0:
// https://github.com/euank/go-kmsg-parser/blob/v3.0.0/kmsgparser/kmsgparser.go
// The upstream Logger abstraction (log.go) is dropped; the parser logs through
// klog instead.
package kmsgparser

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	"k8s.io/klog/v2"
)

// Parser is a parser for the kernel ring buffer found at /dev/kmsg
type Parser interface {
	// SeekEnd moves the parser to the end of the kmsg queue.
	SeekEnd() error
	// Parse reads from kmsg and provides a channel of messages.
	// Parse will always close the provided channel before returning.
	// Parse may be canceled by calling 'Close' on the parser.
	//
	// The caller should drain the channel after calling [Close]. The caller must
	// not close the channel passed in.
	Parse(chan<- Message) error

	Close() error
}

// Message represents a given kmsg logline, including its timestamp (as
// calculated based on offset from boot time), its possibly multi-line body,
// and so on. More information about these mssages may be found here:
// https://www.kernel.org/doc/Documentation/ABI/testing/dev-kmsg
type Message struct {
	Priority       int
	SequenceNumber int
	Timestamp      time.Time
	Message        string
}

// NewParser constructs a new parser with the given Options.
func NewParser(opts ...Option) (Parser, error) {
	f, err := os.Open("/dev/kmsg")
	if err != nil {
		return nil, err
	}

	bootTime, err := getBootTime()
	if err != nil {
		return nil, err
	}
	p := &parser{
		kmsgReader: f,
		bootTime:   bootTime,
		follow:     true,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p, nil
}

// Option is a configuration option for [NewParser]
type Option func(p *parser)

// WithNoFollow configures [Parser] to stop reading and close the channel when
// it reaches the end of the kmsg buffer.
func WithNoFollow() func(p *parser) {
	return func(p *parser) {
		p.follow = false
	}
}

type parser struct {
	kmsgReader *os.File
	bootTime   time.Time
	// follow indicates whether we should stop when we hit the end of the kmsg
	// buffer, or keep reading. Similar to dmesg --follow.
	// Default true
	follow bool
}

func getBootTime() (time.Time, error) {
	var sysinfo syscall.Sysinfo_t
	err := syscall.Sysinfo(&sysinfo)
	if err != nil {
		return time.Time{}, fmt.Errorf("could not get boot time: %v", err)
	}
	// sysinfo only has seconds
	return time.Now().Add(-1 * (time.Duration(sysinfo.Uptime) * time.Second)), nil
}

func (p *parser) SeekEnd() error {
	_, err := p.kmsgReader.Seek(0, io.SeekEnd)
	return err
}

func (p *parser) Close() error {
	return p.kmsgReader.Close()
}

// Parse reads from the provided reader and provide a channel of messages
// parsed.
// If the provided reader *is not* a proper Linux kmsg device, Parse might not
// behave correctly since it relies on specific behavior of `/dev/kmsg`
//
// The caller should typically run 'Parse' in a goroutine.
func (p *parser) Parse(msgs chan<- Message) error {
	// with follow (dmesg --follow), we can use go's usual way of reading thing (i.e. epoll + nonblocking IO).
	if p.follow {
		return p.readFollow(msgs)
	}
	return p.readNofollow(msgs)
}

func (p *parser) readFollow(msgs chan<- Message) error {
	defer close(msgs)
	msg := make([]byte, 8192)
	for {
		// Each read call gives us one full message.
		// https://www.kernel.org/doc/Documentation/ABI/testing/dev-kmsg
		n, err := p.kmsgReader.Read(msg)
		switch {
		case err == nil:
		case errors.Is(err, syscall.EPIPE):
			klog.Warningf("short read from kmsg; skipping")
			continue
		case errors.Is(err, io.EOF), errors.Is(err, os.ErrClosed):
			// someone closed us
			return nil
		default:
			return fmt.Errorf("unexpected kmsg reading error: %w", err)
		}
		msgStr := string(msg[:n])
		message, err := p.parseMessage(msgStr)
		if err != nil {
			return fmt.Errorf("malformed kmsg message: %w", err)
		}
		msgs <- message
	}
}

func (p *parser) readNofollow(msgs chan<- Message) error {
	defer close(msgs)

	rawReader, err := p.kmsgReader.SyscallConn()
	if err != nil {
		return err
	}
	// we're forced to put the fd into non-blocking mode to be able to detect the
	// end of the buffer, but to not use go's built-in epoll
	if ctrlErr := rawReader.Control(func(fd uintptr) {
		err = syscall.SetNonblock(int(fd), true)
	}); ctrlErr != nil {
		return fmt.Errorf("error calling control on kmsg reader: %w", ctrlErr)
	}
	if err != nil {
		return fmt.Errorf("unable to set nonblocking on fd: %w", err)
	}
	msg := make([]byte, 8192)
	for {
		// Each read call gives us one full message.
		// https://www.kernel.org/doc/Documentation/ABI/testing/dev-kmsg
		var err error
		var n int
		readErr := rawReader.Read(func(fd uintptr) bool {
			n, err = syscall.Read(int(fd), msg)
			return true
		})
		if readErr != nil && err == nil {
			err = readErr
		}
		switch {
		case err == nil:
		case errors.Is(err, syscall.EPIPE):
			klog.Warningf("short read from kmsg; skipping")
			continue
		case errors.Is(err, syscall.EAGAIN):
			// end of ring buffer in nofollow mode, we're done
			return nil
		default:
			return fmt.Errorf("unexpected kmsg reading error: %w", err)
		}
		msgStr := string(msg[:n])

		message, err := p.parseMessage(msgStr)
		if err != nil {
			return fmt.Errorf("malformed kmsg message: %w", err)
		}
		msgs <- message
	}
}

func (p *parser) parseMessage(input string) (Message, error) {
	// Format:
	//   PRIORITY,SEQUENCE_NUM,TIMESTAMP,-;MESSAGE
	parts := strings.SplitN(input, ";", 2)
	if len(parts) != 2 {
		return Message{}, fmt.Errorf("invalid kmsg; must contain a ';'")
	}

	metadata, message := parts[0], parts[1]

	metadataParts := strings.Split(metadata, ",")
	if len(metadataParts) < 3 {
		return Message{}, fmt.Errorf("invalid kmsg: must contain at least 3 ',' separated pieces at the start")
	}

	priority, sequence, timestamp := metadataParts[0], metadataParts[1], metadataParts[2]

	prioNum, err := strconv.Atoi(priority)
	if err != nil {
		return Message{}, fmt.Errorf("could not parse %q as priority: %v", priority, err)
	}

	sequenceNum, err := strconv.Atoi(sequence)
	if err != nil {
		return Message{}, fmt.Errorf("could not parse %q as sequence number: %v", priority, err)
	}

	timestampUsFromBoot, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return Message{}, fmt.Errorf("could not parse %q as timestamp: %v", priority, err)
	}
	// timestamp is offset in microsecond from boottime.
	msgTime := p.bootTime.Add(time.Duration(timestampUsFromBoot) * time.Microsecond)

	return Message{
		Priority:       prioNum,
		SequenceNumber: sequenceNum,
		Timestamp:      msgTime,
		Message:        message,
	}, nil
}
