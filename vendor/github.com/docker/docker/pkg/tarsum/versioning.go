package tarsum

import (
	"archive/tar"
	"errors"
	"sort"
	"strconv"
	"strings"
)

// versioning of the TarSum algorithm
// based on the prefix of the hash used
// i.e. "tarsum+sha256:e58fcf7418d4390dec8e8fb69d88c06ec07039d651fedd3aa72af9972e7d046b"
type Version int

// Prefix of "tarsum"
const (
	Version0 Version = iota
	Version1
	// NOTE: this variable will be either the latest or an unsettled next-version of the TarSum calculation
	VersionDev
)

// VersionLabelForChecksum returns the label for the given tarsum
// checksum, i.e., everything before the first `+` character in
// the string or an empty string if no label separator is found.
func VersionLabelForChecksum(checksum string) string {
	// Checksums are in the form: {versionLabel}+{hashID}:{hex}
	sepIndex := strings.Index(checksum, "+")
	if sepIndex < 0 {
		return ""
	}
	return checksum[:sepIndex]
}

// Get a list of all known tarsum Version
func GetVersions() []Version {
	v := []Version{}
	for k := range tarSumVersions {
		v = append(v, k)
	}
	return v
}

var (
	tarSumVersions = map[Version]string{
		Version0:   "tarsum",
		Version1:   "tarsum.v1",
		VersionDev: "tarsum.dev",
	}
	tarSumVersionsByName = map[string]Version{
		"tarsum":     Version0,
		"tarsum.v1":  Version1,
		"tarsum.dev": VersionDev,
	}
)

func (tsv Version) String() string {
	return tarSumVersions[tsv]
}

// GetVersionFromTarsum returns the Version from the provided string
func GetVersionFromTarsum(tarsum string) (Version, error) {
	tsv := tarsum
	if strings.Contains(tarsum, "+") {
		tsv = strings.SplitN(tarsum, "+", 2)[0]
	}
	for v, s := range tarSumVersions {
		if s == tsv {
			return v, nil
		}
	}
	return -1, ErrNotVersion
}

// Errors that may be returned by functions in this package
var (
	ErrNotVersion            = errors.New("string does not include a TarSum Version")
	ErrVersionNotImplemented = errors.New("TarSum Version is not yet implemented")
)

// tarHeaderSelector is the interface which different versions
// of tarsum should use for selecting and ordering tar headers
// for each item in the archive.
type tarHeaderSelector interface {
	selectHeaders(h *tar.Header) (orderedHeaders [][2]string)
}

type tarHeaderSelectFunc func(h *tar.Header) (orderedHeaders [][2]string)

func (f tarHeaderSelectFunc) selectHeaders(h *tar.Header) (orderedHeaders [][2]string) {
	return f(h)
}

func v0TarHeaderSelect(h *tar.Header) (orderedHeaders [][2]string) {
	return [][2]string{
		{"name", h.Name},
		{"mode", strconv.Itoa(int(h.Mode))},
		{"uid", strconv.Itoa(h.Uid)},
		{"gid", strconv.Itoa(h.Gid)},
		{"size", strconv.Itoa(int(h.Size))},
		{"mtime", strconv.Itoa(int(h.ModTime.UTC().Unix()))},
		{"typeflag", string([]byte{h.Typeflag})},
		{"linkname", h.Linkname},
		{"uname", h.Uname},
		{"gname", h.Gname},
		{"devmajor", strconv.Itoa(int(h.Devmajor))},
		{"devminor", strconv.Itoa(int(h.Devminor))},
	}
}

func v1TarHeaderSelect(h *tar.Header) (orderedHeaders [][2]string) {
	// Get extended attributes.
	xAttrKeys := make([]string, len(h.Xattrs))
	for k := range h.Xattrs {
		xAttrKeys = append(xAttrKeys, k)
	}
	sort.Strings(xAttrKeys)

	// Make the slice with enough capacity to hold the 11 basic headers
	// we want from the v0 selector plus however many xattrs we have.
	orderedHeaders = make([][2]string, 0, 11+len(xAttrKeys))

	// Copy all headers from v0 excluding the 'mtime' header (the 5th element).
	v0headers := v0TarHeaderSelect(h)
	orderedHeaders = append(orderedHeaders, v0headers[0:5]...)
	orderedHeaders = append(orderedHeaders, v0headers[6:]...)

	// Finally, append the sorted xattrs.
	for _, k := range xAttrKeys {
		orderedHeaders = append(orderedHeaders, [2]string{k, h.Xattrs[k]})
	}

	return
}

var registeredHeaderSelectors = map[Version]tarHeaderSelectFunc{
	Version0:   v0TarHeaderSelect,
	Version1:   v1TarHeaderSelect,
	VersionDev: v1TarHeaderSelect,
}

func getTarHeaderSelector(v Version) (tarHeaderSelector, error) {
	headerSelector, ok := registeredHeaderSelectors[v]
	if !ok {
		return nil, ErrVersionNotImplemented
	}

	return headerSelector, nil
}
