package intelrdt

import (
	"bufio"
	"github.com/sirupsen/logrus"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

var (
	enabledMonFeatures monFeatures
)

type monFeatures struct {
	mbmTotalBytes bool
	mbmLocalBytes bool
	llcOccupancy  bool
}

func getMonFeatures(intelRdtRoot string) (monFeatures, error) {
	file, err := os.Open(filepath.Join(intelRdtRoot, "info", "L3_MON", "mon_features"))
	defer file.Close()
	if err != nil {
		return monFeatures{}, err
	}
	return parseMonFeatures(file)
}

func parseMonFeatures(reader io.Reader) (monFeatures, error) {
	scanner := bufio.NewScanner(reader)

	monFeatures := monFeatures{}

	for scanner.Scan() {
		switch feature := scanner.Text(); feature {
		case "mbm_total_bytes":
			monFeatures.mbmTotalBytes = true
		case "mbm_local_bytes":
			monFeatures.mbmLocalBytes = true
		case "llc_occupancy":
			monFeatures.llcOccupancy = true
		default:
			logrus.Warnf("Unsupported Intel RDT monitoring feature: %s", feature)
		}
	}

	return monFeatures, scanner.Err()
}

func getMonitoringStats(containerPath string, stats *Stats) error {
	numaFiles, err := ioutil.ReadDir(filepath.Join(containerPath, "mon_data"))
	if err != nil {
		return err
	}

	var mbmStats []MBMNumaNodeStats
	var cmtStats []CMTNumaNodeStats

	for _, file := range numaFiles {
		if file.IsDir() {
			numaPath := filepath.Join(containerPath, "mon_data", file.Name())
			if IsMBMEnabled() {
				numaMBMStats, err := getMBMNumaNodeStats(numaPath)
				if err != nil {
					return err
				}
				mbmStats = append(mbmStats, *numaMBMStats)
			}
			if IsCMTEnabled() {
				numaCMTStats, err := getCMTNumaNodeStats(numaPath)
				if err != nil {
					return err
				}
				cmtStats = append(cmtStats, *numaCMTStats)
			}
		}
	}

	stats.MBMStats = &mbmStats
	stats.CMTStats = &cmtStats

	return err
}
