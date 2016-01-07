package opts

import (
	"fmt"
	"net"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/parsers"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/volume"
)

var (
	alphaRegexp  = regexp.MustCompile(`[a-zA-Z]`)
	domainRegexp = regexp.MustCompile(`^(:?(:?[a-zA-Z0-9]|(:?[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]))(:?\.(:?[a-zA-Z0-9]|(:?[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])))*)\.?\s*$`)
	// DefaultHTTPHost Default HTTP Host used if only port is provided to -H flag e.g. docker -d -H tcp://:8080
	DefaultHTTPHost = "127.0.0.1"
	// DefaultHTTPPort Default HTTP Port used if only the protocol is provided to -H flag e.g. docker -d -H tcp://
	// TODO Windows. DefaultHTTPPort is only used on Windows if a -H parameter
	// is not supplied. A better longer term solution would be to use a named
	// pipe as the default on the Windows daemon.
	DefaultHTTPPort = 2375 // Default HTTP Port
	// DefaultUnixSocket Path for the unix socket.
	// Docker daemon by default always listens on the default unix socket
	DefaultUnixSocket = "/var/run/docker.sock"
)

// ListOpts type that hold a list of values and a validation function.
type ListOpts struct {
	values    *[]string
	validator ValidatorFctType
}

// NewListOpts Create a new ListOpts with the specified validator.
func NewListOpts(validator ValidatorFctType) ListOpts {
	var values []string
	return *NewListOptsRef(&values, validator)
}

func NewListOptsRef(values *[]string, validator ValidatorFctType) *ListOpts {
	return &ListOpts{
		values:    values,
		validator: validator,
	}
}

func (opts *ListOpts) String() string {
	return fmt.Sprintf("%v", []string((*opts.values)))
}

// Set validates if needed the input value and add it to the
// internal slice.
func (opts *ListOpts) Set(value string) error {
	if opts.validator != nil {
		v, err := opts.validator(value)
		if err != nil {
			return err
		}
		value = v
	}
	(*opts.values) = append((*opts.values), value)
	return nil
}

// Delete remove the given element from the slice.
func (opts *ListOpts) Delete(key string) {
	for i, k := range *opts.values {
		if k == key {
			(*opts.values) = append((*opts.values)[:i], (*opts.values)[i+1:]...)
			return
		}
	}
}

// GetMap returns the content of values in a map in order to avoid
// duplicates.
// FIXME: can we remove this?
func (opts *ListOpts) GetMap() map[string]struct{} {
	ret := make(map[string]struct{})
	for _, k := range *opts.values {
		ret[k] = struct{}{}
	}
	return ret
}

// GetAll returns the values' slice.
// FIXME: Can we remove this?
func (opts *ListOpts) GetAll() []string {
	return (*opts.values)
}

// Get checks the existence of the given key.
func (opts *ListOpts) Get(key string) bool {
	for _, k := range *opts.values {
		if k == key {
			return true
		}
	}
	return false
}

// Len returns the amount of element in the slice.
func (opts *ListOpts) Len() int {
	return len((*opts.values))
}

//MapOpts type that holds a map of values and a validation function.
type MapOpts struct {
	values    map[string]string
	validator ValidatorFctType
}

// Set validates if needed the input value and add it to the
// internal map, by splitting on '='.
func (opts *MapOpts) Set(value string) error {
	if opts.validator != nil {
		v, err := opts.validator(value)
		if err != nil {
			return err
		}
		value = v
	}
	vals := strings.SplitN(value, "=", 2)
	if len(vals) == 1 {
		(opts.values)[vals[0]] = ""
	} else {
		(opts.values)[vals[0]] = vals[1]
	}
	return nil
}

func (opts *MapOpts) String() string {
	return fmt.Sprintf("%v", map[string]string((opts.values)))
}

func NewMapOpts(values map[string]string, validator ValidatorFctType) *MapOpts {
	if values == nil {
		values = make(map[string]string)
	}
	return &MapOpts{
		values:    values,
		validator: validator,
	}
}

// ValidatorFctType validator that return a validate string and/or an error
type ValidatorFctType func(val string) (string, error)

// ValidatorFctListType validator that return a validate list of string and/or an error
type ValidatorFctListType func(val string) ([]string, error)

// ValidateAttach Validates that the specified string is a valid attach option.
func ValidateAttach(val string) (string, error) {
	s := strings.ToLower(val)
	for _, str := range []string{"stdin", "stdout", "stderr"} {
		if s == str {
			return s, nil
		}
	}
	return val, fmt.Errorf("valid streams are STDIN, STDOUT and STDERR")
}

// ValidateLink Validates that the specified string has a valid link format (containerName:alias).
func ValidateLink(val string) (string, error) {
	if _, _, err := parsers.ParseLink(val); err != nil {
		return val, err
	}
	return val, nil
}

// ValidateDevice Validate a path for devices
// It will make sure 'val' is in the form:
//    [host-dir:]container-path[:mode]
func ValidateDevice(val string) (string, error) {
	return validatePath(val, false)
}

// ValidatePath Validate a path for volumes
// It will make sure 'val' is in the form:
//    [host-dir:]container-path[:rw|ro]
// It will also validate the mount mode.
func ValidatePath(val string) (string, error) {
	return validatePath(val, true)
}

func validatePath(val string, validateMountMode bool) (string, error) {
	var containerPath string
	var mode string

	if strings.Count(val, ":") > 2 {
		return val, fmt.Errorf("bad format for volumes: %s", val)
	}

	splited := strings.SplitN(val, ":", 3)
	if splited[0] == "" {
		return val, fmt.Errorf("bad format for volumes: %s", val)
	}
	switch len(splited) {
	case 1:
		containerPath = splited[0]
		val = path.Clean(containerPath)
	case 2:
		if isValid, _ := volume.ValidateMountMode(splited[1]); validateMountMode && isValid {
			containerPath = splited[0]
			mode = splited[1]
			val = fmt.Sprintf("%s:%s", path.Clean(containerPath), mode)
		} else {
			containerPath = splited[1]
			val = fmt.Sprintf("%s:%s", splited[0], path.Clean(containerPath))
		}
	case 3:
		containerPath = splited[1]
		mode = splited[2]
		if isValid, _ := volume.ValidateMountMode(splited[2]); validateMountMode && !isValid {
			return val, fmt.Errorf("bad mount mode specified : %s", mode)
		}
		val = fmt.Sprintf("%s:%s:%s", splited[0], containerPath, mode)
	}

	if !path.IsAbs(containerPath) {
		return val, fmt.Errorf("%s is not an absolute path", containerPath)
	}
	return val, nil
}

// ValidateEnv Validate an environment variable and returns it
// It will use EnvironmentVariableRegexp to ensure the name of the environment variable is valid.
// If no value is specified, it returns the current value using os.Getenv.
func ValidateEnv(val string) (string, error) {
	arr := strings.Split(val, "=")
	if len(arr) > 1 {
		return val, nil
	}
	if !EnvironmentVariableRegexp.MatchString(arr[0]) {
		return val, ErrBadEnvVariable{fmt.Sprintf("variable '%s' is not a valid environment variable", val)}
	}
	if !doesEnvExist(val) {
		return val, nil
	}
	return fmt.Sprintf("%s=%s", val, os.Getenv(val)), nil
}

// ValidateIPAddress Validates an Ip address
func ValidateIPAddress(val string) (string, error) {
	var ip = net.ParseIP(strings.TrimSpace(val))
	if ip != nil {
		return ip.String(), nil
	}
	return "", fmt.Errorf("%s is not an ip address", val)
}

// ValidateMACAddress Validates a MAC address
func ValidateMACAddress(val string) (string, error) {
	_, err := net.ParseMAC(strings.TrimSpace(val))
	if err != nil {
		return "", err
	}
	return val, nil
}

// ValidateDNSSearch Validates domain for resolvconf search configuration.
// A zero length domain is represented by .
func ValidateDNSSearch(val string) (string, error) {
	if val = strings.Trim(val, " "); val == "." {
		return val, nil
	}
	return validateDomain(val)
}

func validateDomain(val string) (string, error) {
	if alphaRegexp.FindString(val) == "" {
		return "", fmt.Errorf("%s is not a valid domain", val)
	}
	ns := domainRegexp.FindSubmatch([]byte(val))
	if len(ns) > 0 && len(ns[1]) < 255 {
		return string(ns[1]), nil
	}
	return "", fmt.Errorf("%s is not a valid domain", val)
}

// ValidateExtraHost Validate that the given string is a valid extrahost and returns it
// ExtraHost are in the form of name:ip where the ip has to be a valid ip (ipv4 or ipv6)
func ValidateExtraHost(val string) (string, error) {
	// allow for IPv6 addresses in extra hosts by only splitting on first ":"
	arr := strings.SplitN(val, ":", 2)
	if len(arr) != 2 || len(arr[0]) == 0 {
		return "", fmt.Errorf("bad format for add-host: %q", val)
	}
	if _, err := ValidateIPAddress(arr[1]); err != nil {
		return "", fmt.Errorf("invalid IP address in add-host: %q", arr[1])
	}
	return val, nil
}

// ValidateLabel Validate that the given string is a valid label, and returns it
// Labels are in the form on key=value
func ValidateLabel(val string) (string, error) {
	if strings.Count(val, "=") < 1 {
		return "", fmt.Errorf("bad attribute format: %s", val)
	}
	return val, nil
}

// ValidateHost Validate that the given string is a valid host and returns it
func ValidateHost(val string) (string, error) {
	host, err := parsers.ParseHost(DefaultHTTPHost, DefaultUnixSocket, val)
	if err != nil {
		return val, err
	}
	return host, nil
}

func doesEnvExist(name string) bool {
	for _, entry := range os.Environ() {
		parts := strings.SplitN(entry, "=", 2)
		if parts[0] == name {
			return true
		}
	}
	return false
}
