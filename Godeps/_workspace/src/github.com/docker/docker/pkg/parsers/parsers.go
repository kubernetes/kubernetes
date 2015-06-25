package parsers

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"
)

// FIXME: Change this not to receive default value as parameter
func ParseHost(defaultTCPAddr, defaultUnixAddr, addr string) (string, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		if runtime.GOOS != "windows" {
			addr = fmt.Sprintf("unix://%s", defaultUnixAddr)
		} else {
			// Note - defaultTCPAddr already includes tcp:// prefix
			addr = fmt.Sprintf("%s", defaultTCPAddr)
		}
	}
	addrParts := strings.Split(addr, "://")
	if len(addrParts) == 1 {
		addrParts = []string{"tcp", addrParts[0]}
	}

	switch addrParts[0] {
	case "tcp":
		return ParseTCPAddr(addrParts[1], defaultTCPAddr)
	case "unix":
		return ParseUnixAddr(addrParts[1], defaultUnixAddr)
	case "fd":
		return addr, nil
	default:
		return "", fmt.Errorf("Invalid bind address format: %s", addr)
	}
}

func ParseUnixAddr(addr string, defaultAddr string) (string, error) {
	addr = strings.TrimPrefix(addr, "unix://")
	if strings.Contains(addr, "://") {
		return "", fmt.Errorf("Invalid proto, expected unix: %s", addr)
	}
	if addr == "" {
		addr = defaultAddr
	}
	return fmt.Sprintf("unix://%s", addr), nil
}

func ParseTCPAddr(addr string, defaultAddr string) (string, error) {
	addr = strings.TrimPrefix(addr, "tcp://")
	if strings.Contains(addr, "://") || addr == "" {
		return "", fmt.Errorf("Invalid proto, expected tcp: %s", addr)
	}

	hostParts := strings.Split(addr, ":")
	if len(hostParts) != 2 {
		return "", fmt.Errorf("Invalid bind address format: %s", addr)
	}
	host := hostParts[0]
	if host == "" {
		host = defaultAddr
	}

	p, err := strconv.Atoi(hostParts[1])
	if err != nil && p == 0 {
		return "", fmt.Errorf("Invalid bind address format: %s", addr)
	}
	return fmt.Sprintf("tcp://%s:%d", host, p), nil
}

// Get a repos name and returns the right reposName + tag|digest
// The tag can be confusing because of a port in a repository name.
//     Ex: localhost.localdomain:5000/samalba/hipache:latest
//     Digest ex: localhost:5000/foo/bar@sha256:bc8813ea7b3603864987522f02a76101c17ad122e1c46d790efc0fca78ca7bfb
func ParseRepositoryTag(repos string) (string, string) {
	n := strings.Index(repos, "@")
	if n >= 0 {
		parts := strings.Split(repos, "@")
		return parts[0], parts[1]
	}
	n = strings.LastIndex(repos, ":")
	if n < 0 {
		return repos, ""
	}
	if tag := repos[n+1:]; !strings.Contains(tag, "/") {
		return repos[:n], tag
	}
	return repos, ""
}

func PartParser(template, data string) (map[string]string, error) {
	// ip:public:private
	var (
		templateParts = strings.Split(template, ":")
		parts         = strings.Split(data, ":")
		out           = make(map[string]string, len(templateParts))
	)
	if len(parts) != len(templateParts) {
		return nil, fmt.Errorf("Invalid format to parse.  %s should match template %s", data, template)
	}

	for i, t := range templateParts {
		value := ""
		if len(parts) > i {
			value = parts[i]
		}
		out[t] = value
	}
	return out, nil
}

func ParseKeyValueOpt(opt string) (string, string, error) {
	parts := strings.SplitN(opt, "=", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("Unable to parse key/value option: %s", opt)
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), nil
}

func ParsePortRange(ports string) (uint64, uint64, error) {
	if ports == "" {
		return 0, 0, fmt.Errorf("Empty string specified for ports.")
	}
	if !strings.Contains(ports, "-") {
		start, err := strconv.ParseUint(ports, 10, 16)
		end := start
		return start, end, err
	}

	parts := strings.Split(ports, "-")
	start, err := strconv.ParseUint(parts[0], 10, 16)
	if err != nil {
		return 0, 0, err
	}
	end, err := strconv.ParseUint(parts[1], 10, 16)
	if err != nil {
		return 0, 0, err
	}
	if end < start {
		return 0, 0, fmt.Errorf("Invalid range specified for the Port: %s", ports)
	}
	return start, end, nil
}

func ParseLink(val string) (string, string, error) {
	if val == "" {
		return "", "", fmt.Errorf("empty string specified for links")
	}
	arr := strings.Split(val, ":")
	if len(arr) > 2 {
		return "", "", fmt.Errorf("bad format for links: %s", val)
	}
	if len(arr) == 1 {
		return val, val, nil
	}
	return arr[0], arr[1], nil
}
