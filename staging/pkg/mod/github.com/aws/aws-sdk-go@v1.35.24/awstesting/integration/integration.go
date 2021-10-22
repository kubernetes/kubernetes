// +build integration

// Package integration performs initialization and validation for integration
// tests.
package integration

import (
	"crypto/rand"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
)

// Session is a shared session for all integration tests to use.
var Session = session.Must(session.NewSession(&aws.Config{
	CredentialsChainVerboseErrors: aws.Bool(true),
}))

func init() {
	logLevel := Session.Config.LogLevel
	if os.Getenv("DEBUG") != "" {
		logLevel = aws.LogLevel(aws.LogDebug)
	}
	if os.Getenv("DEBUG_SIGNING") != "" {
		logLevel = aws.LogLevel(aws.LogDebugWithSigning)
	}
	if os.Getenv("DEBUG_BODY") != "" {
		logLevel = aws.LogLevel(aws.LogDebugWithSigning | aws.LogDebugWithHTTPBody)
	}
	Session.Config.LogLevel = logLevel
}

// UniqueID returns a unique UUID-like identifier for use in generating
// resources for integration tests.
func UniqueID() string {
	uuid := make([]byte, 16)
	io.ReadFull(rand.Reader, uuid)
	return fmt.Sprintf("%x", uuid)
}

// CreateFileOfSize will return an *os.File that is of size bytes
func CreateFileOfSize(dir string, size int64) (*os.File, error) {
	file, err := ioutil.TempFile(dir, "s3Bench")
	if err != nil {
		return nil, err
	}

	err = file.Truncate(size)
	if err != nil {
		file.Close()
		os.Remove(file.Name())
		return nil, err
	}

	return file, nil
}

// SizeToName returns a human-readable string for the given size bytes
func SizeToName(size int) string {
	units := []string{"B", "KB", "MB", "GB"}
	i := 0
	for size >= 1024 {
		size /= 1024
		i++
	}

	if i > len(units)-1 {
		i = len(units) - 1
	}

	return fmt.Sprintf("%d%s", size, units[i])
}

// SessionWithDefaultRegion returns a copy of the integration session with the
// region set if one was not already provided.
func SessionWithDefaultRegion(region string) *session.Session {
	sess := Session.Copy()
	if v := aws.StringValue(sess.Config.Region); len(v) == 0 {
		sess.Config.Region = aws.String(region)
	}

	return sess
}
