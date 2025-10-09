// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"errors"
	"strings"

	semconv "go.opentelemetry.io/otel/semconv/v1.34.0"
)

type hostIDProvider func() (string, error)

var defaultHostIDProvider hostIDProvider = platformHostIDReader.read

var hostID = defaultHostIDProvider

type hostIDReader interface {
	read() (string, error)
}

type fileReader func(string) (string, error)

type commandExecutor func(string, ...string) (string, error)

// hostIDReaderBSD implements hostIDReader.
type hostIDReaderBSD struct {
	execCommand commandExecutor
	readFile    fileReader
}

// read attempts to read the machine-id from /etc/hostid. If not found it will
// execute `kenv -q smbios.system.uuid`. If neither location yields an id an
// error will be returned.
func (r *hostIDReaderBSD) read() (string, error) {
	if result, err := r.readFile("/etc/hostid"); err == nil {
		return strings.TrimSpace(result), nil
	}

	if result, err := r.execCommand("kenv", "-q", "smbios.system.uuid"); err == nil {
		return strings.TrimSpace(result), nil
	}

	return "", errors.New("host id not found in: /etc/hostid or kenv")
}

// hostIDReaderDarwin implements hostIDReader.
type hostIDReaderDarwin struct {
	execCommand commandExecutor
}

// read executes `ioreg -rd1 -c "IOPlatformExpertDevice"` and parses host id
// from the IOPlatformUUID line. If the command fails or the uuid cannot be
// parsed an error will be returned.
func (r *hostIDReaderDarwin) read() (string, error) {
	result, err := r.execCommand("ioreg", "-rd1", "-c", "IOPlatformExpertDevice")
	if err != nil {
		return "", err
	}

	lines := strings.Split(result, "\n")
	for _, line := range lines {
		if strings.Contains(line, "IOPlatformUUID") {
			parts := strings.Split(line, " = ")
			if len(parts) == 2 {
				return strings.Trim(parts[1], "\""), nil
			}
			break
		}
	}

	return "", errors.New("could not parse IOPlatformUUID")
}

type hostIDReaderLinux struct {
	readFile fileReader
}

// read attempts to read the machine-id from /etc/machine-id followed by
// /var/lib/dbus/machine-id. If neither location yields an ID an error will
// be returned.
func (r *hostIDReaderLinux) read() (string, error) {
	if result, err := r.readFile("/etc/machine-id"); err == nil {
		return strings.TrimSpace(result), nil
	}

	if result, err := r.readFile("/var/lib/dbus/machine-id"); err == nil {
		return strings.TrimSpace(result), nil
	}

	return "", errors.New("host id not found in: /etc/machine-id or /var/lib/dbus/machine-id")
}

type hostIDDetector struct{}

// Detect returns a *Resource containing the platform specific host id.
func (hostIDDetector) Detect(ctx context.Context) (*Resource, error) {
	hostID, err := hostID()
	if err != nil {
		return nil, err
	}

	return NewWithAttributes(
		semconv.SchemaURL,
		semconv.HostID(hostID),
	), nil
}
