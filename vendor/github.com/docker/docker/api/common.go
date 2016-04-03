package api

import (
	"fmt"
	"mime"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/version"
	"github.com/docker/libtrust"
)

// Common constants for daemon and client.
const (
	// Current REST API version
	Version version.Version = "1.20"

	// Minimun REST API version supported
	MinVersion version.Version = "1.12"

	// Default filename with Docker commands, read by docker build
	DefaultDockerfileName string = "Dockerfile"
)

type ByPrivatePort []types.Port

func (r ByPrivatePort) Len() int           { return len(r) }
func (r ByPrivatePort) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r ByPrivatePort) Less(i, j int) bool { return r[i].PrivatePort < r[j].PrivatePort }

func DisplayablePorts(ports []types.Port) string {
	var (
		result          = []string{}
		hostMappings    = []string{}
		firstInGroupMap map[string]int
		lastInGroupMap  map[string]int
	)
	firstInGroupMap = make(map[string]int)
	lastInGroupMap = make(map[string]int)
	sort.Sort(ByPrivatePort(ports))
	for _, port := range ports {
		var (
			current      = port.PrivatePort
			portKey      = port.Type
			firstInGroup int
			lastInGroup  int
		)
		if port.IP != "" {
			if port.PublicPort != current {
				hostMappings = append(hostMappings, fmt.Sprintf("%s:%d->%d/%s", port.IP, port.PublicPort, port.PrivatePort, port.Type))
				continue
			}
			portKey = fmt.Sprintf("%s/%s", port.IP, port.Type)
		}
		firstInGroup = firstInGroupMap[portKey]
		lastInGroup = lastInGroupMap[portKey]

		if firstInGroup == 0 {
			firstInGroupMap[portKey] = current
			lastInGroupMap[portKey] = current
			continue
		}

		if current == (lastInGroup + 1) {
			lastInGroupMap[portKey] = current
			continue
		}
		result = append(result, FormGroup(portKey, firstInGroup, lastInGroup))
		firstInGroupMap[portKey] = current
		lastInGroupMap[portKey] = current
	}
	for portKey, firstInGroup := range firstInGroupMap {
		result = append(result, FormGroup(portKey, firstInGroup, lastInGroupMap[portKey]))
	}
	result = append(result, hostMappings...)
	return strings.Join(result, ", ")
}

func FormGroup(key string, start, last int) string {
	var (
		group     string
		parts     = strings.Split(key, "/")
		groupType = parts[0]
		ip        = ""
	)
	if len(parts) > 1 {
		ip = parts[0]
		groupType = parts[1]
	}
	if start == last {
		group = fmt.Sprintf("%d", start)
	} else {
		group = fmt.Sprintf("%d-%d", start, last)
	}
	if ip != "" {
		group = fmt.Sprintf("%s:%s->%s", ip, group, group)
	}
	return fmt.Sprintf("%s/%s", group, groupType)
}

func MatchesContentType(contentType, expectedType string) bool {
	mimetype, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		logrus.Errorf("Error parsing media type: %s error: %v", contentType, err)
	}
	return err == nil && mimetype == expectedType
}

// LoadOrCreateTrustKey attempts to load the libtrust key at the given path,
// otherwise generates a new one
func LoadOrCreateTrustKey(trustKeyPath string) (libtrust.PrivateKey, error) {
	err := system.MkdirAll(filepath.Dir(trustKeyPath), 0700)
	if err != nil {
		return nil, err
	}
	trustKey, err := libtrust.LoadKeyFile(trustKeyPath)
	if err == libtrust.ErrKeyFileDoesNotExist {
		trustKey, err = libtrust.GenerateECP256PrivateKey()
		if err != nil {
			return nil, fmt.Errorf("Error generating key: %s", err)
		}
		if err := libtrust.SaveKey(trustKeyPath, trustKey); err != nil {
			return nil, fmt.Errorf("Error saving key file: %s", err)
		}
	} else if err != nil {
		return nil, fmt.Errorf("Error loading key file %s: %s", trustKeyPath, err)
	}
	return trustKey, nil
}
