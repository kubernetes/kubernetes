// Copyright (c) 2015 HPE Software Inc. All rights reserved.
// Copyright (c) 2013 ActiveState Software Inc. All rights reserved.

package util

import (
	"fmt"
	"log"
	"os"
	"runtime/debug"
)

type Logger struct {
	*log.Logger
}

var LOGGER = &Logger{log.New(os.Stderr, "", log.LstdFlags)}

// fatal is like panic except it displays only the current goroutine's stack.
func Fatal(format string, v ...interface{}) {
	// https://github.com/nxadm/log/blob/master/log.go#L45
	LOGGER.Output(2, fmt.Sprintf("FATAL -- "+format, v...)+"\n"+string(debug.Stack()))
	os.Exit(1)
}

// partitionString partitions the string into chunks of given size,
// with the last chunk of variable size.
func PartitionString(s string, chunkSize int) []string {
	if chunkSize <= 0 {
		panic("invalid chunkSize")
	}
	length := len(s)
	chunks := 1 + length/chunkSize
	start := 0
	end := chunkSize
	parts := make([]string, 0, chunks)
	for {
		if end > length {
			end = length
		}
		parts = append(parts, s[start:end])
		if end == length {
			break
		}
		start, end = end, end+chunkSize
	}
	return parts
}
