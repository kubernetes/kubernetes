package node

import (
	"fmt"
	"strings"

	"github.com/blang/semver/v4"

	corev1 "k8s.io/api/core/v1"
)

// An error to be returned when kubelet version is lower than specified minimum kubelet version.
// Used to differentiate between a parsing failure, and a failure because the kubelet is out of date.
var ErrKubeletOutdated = fmt.Errorf("kubelet version is outdated")

// ValidateMinimumKubeletVersion takes a list of nodes and a currently set min version.
// It parses the min version and iterates through the nodes, comparing the version of the kubelets
// to the min version.
// It will error if any nodes are older than the min version.
func ValidateMinimumKubeletVersion(nodes []*corev1.Node, minimumKubeletVersion string) error {
	// unset, no error
	if minimumKubeletVersion == "" {
		return nil
	}

	version, err := semver.Parse(minimumKubeletVersion)
	if err != nil {
		return fmt.Errorf("failed to parse submitted version %s %v", minimumKubeletVersion, err.Error())
	}

	for _, node := range nodes {
		if err := IsNodeTooOld(node, &version); err != nil {
			return err
		}
	}
	return nil
}

// IsNodeTooOld answers that very question. It takes a node object and a minVersion,
// parses each into a semver version, and then determines whether the version of the kubelet on the
// node is older than min version.
// When the node is too old, it returns the error ErrKubeletOutdated. If a different error occurs, an error is returned.
// If the node is new enough and no error happens, nil is returned.
func IsNodeTooOld(node *corev1.Node, minVersion *semver.Version) error {
	return IsKubeletVersionTooOld(node.Status.NodeInfo.KubeletVersion, minVersion)
}

// IsKubeletVerisionTooOld answers that very question. It takes a kubelet version and a minVersion,
// parses each into a semver version, and then determines whether the version of the kubelet on the
// node is older than min version.
// It will fail if the minVersion is nil, if the kubeletVersion is invalid, or if the minVersion is greater than
// the kubeletVersion
// When the kubelet is too old, it returns the error ErrKubeletOutdated. If a different error occurs, an error is returned.
// If the node is new enough and no error happens, nil is returned.
func IsKubeletVersionTooOld(kubeletVersion string, minVersion *semver.Version) error {
	if minVersion == nil {
		return fmt.Errorf("given minimum version is nil")
	}
	version, err := ParseKubeletVersion(kubeletVersion)
	if err != nil {
		return fmt.Errorf("failed to parse node version %s: %v", kubeletVersion, err)
	}
	if minVersion.GT(*version) {
		return fmt.Errorf("%w: kubelet version is %v, which is lower than minimumKubeletVersion of %v", ErrKubeletOutdated, *version, *minVersion)
	}
	return nil
}

// ParseKubeletVersion parses it into a semver.Version object, stripping
// any information in the version that isn't "major.minor.patch".
func ParseKubeletVersion(kubeletVersion string) (*semver.Version, error) {
	version, err := semver.Parse(strings.TrimPrefix(kubeletVersion, "v"))
	if err != nil {
		return nil, err
	}

	version.Pre = nil
	version.Build = nil
	return &version, nil
}
