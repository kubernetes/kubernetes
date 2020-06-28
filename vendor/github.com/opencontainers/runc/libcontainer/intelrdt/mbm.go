// +build linux

package intelrdt

var (
	// The flag to indicate if Intel RDT/MBM is enabled
	mbmEnabled bool
)

// Check if Intel RDT/MBM is enabled.
func IsMBMEnabled() bool {
	return mbmEnabled
}

func getMBMNumaNodeStats(numaPath string) (*MBMNumaNodeStats, error) {
	stats := &MBMNumaNodeStats{}
	if enabledMonFeatures.mbmTotalBytes {
		mbmTotalBytes, err := getIntelRdtParamUint(numaPath, "mbm_total_bytes")
		if err != nil {
			return nil, err
		}
		stats.MBMTotalBytes = mbmTotalBytes
	}

	if enabledMonFeatures.mbmLocalBytes {
		mbmLocalBytes, err := getIntelRdtParamUint(numaPath, "mbm_local_bytes")
		if err != nil {
			return nil, err
		}
		stats.MBMLocalBytes = mbmLocalBytes
	}

	return stats, nil
}
