package intelrdt

var (
	cmtEnabled bool
)

// Check if Intel RDT/CMT is enabled.
func IsCMTEnabled() bool {
	featuresInit()
	return cmtEnabled
}

func getCMTNumaNodeStats(numaPath string) (*CMTNumaNodeStats, error) {
	stats := &CMTNumaNodeStats{}

	if enabledMonFeatures.llcOccupancy {
		llcOccupancy, err := getIntelRdtParamUint(numaPath, "llc_occupancy")
		if err != nil {
			return nil, err
		}
		stats.LLCOccupancy = llcOccupancy
	}

	return stats, nil
}
