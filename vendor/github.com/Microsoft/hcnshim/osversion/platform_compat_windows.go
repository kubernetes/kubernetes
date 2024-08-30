package osversion

// List of stable ABI compliant ltsc releases
// Note: List must be sorted in ascending order
var compatLTSCReleases = []uint16{
	V21H2Server,
}

// CheckHostAndContainerCompat checks if given host and container
// OS versions are compatible.
// It includes support for stable ABI compliant versions as well.
// Every release after WS 2022 will support the previous ltsc
// container image. Stable ABI is in preview mode for windows 11 client.
// Refer: https://learn.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility?tabs=windows-server-2022%2Cwindows-10#windows-server-host-os-compatibility
func CheckHostAndContainerCompat(host, ctr OSVersion) bool {
	// check major minor versions of host and guest
	if host.MajorVersion != ctr.MajorVersion ||
		host.MinorVersion != ctr.MinorVersion {
		return false
	}

	// If host is < WS 2022, exact version match is required
	if host.Build < V21H2Server {
		return host.Build == ctr.Build
	}

	var supportedLtscRelease uint16
	for i := len(compatLTSCReleases) - 1; i >= 0; i-- {
		if host.Build >= compatLTSCReleases[i] {
			supportedLtscRelease = compatLTSCReleases[i]
			break
		}
	}
	return ctr.Build >= supportedLtscRelease && ctr.Build <= host.Build
}
