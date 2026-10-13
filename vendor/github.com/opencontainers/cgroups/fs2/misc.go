package fs2

import (
	"bufio"
	"os"
	"strings"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fscommon"
)

func statMisc(dirPath string, stats *cgroups.Stats) error {
	for _, file := range []string{"current", "events"} {
		fd, err := cgroups.OpenFile(dirPath, "misc."+file, os.O_RDONLY)
		if err != nil {
			return err
		}

		s := bufio.NewScanner(fd)
		for s.Scan() {
			key, value, err := fscommon.ParseKeyValue(s.Text())
			if err != nil {
				fd.Close()
				return err
			}

			key = strings.TrimSuffix(key, ".max")

			if _, ok := stats.MiscStats[key]; !ok {
				stats.MiscStats[key] = cgroups.MiscStats{}
			}

			tmp := stats.MiscStats[key]

			switch file {
			case "current":
				tmp.Usage = value
			case "events":
				tmp.Events = value
			}

			stats.MiscStats[key] = tmp
		}
		fd.Close()

		if err := s.Err(); err != nil {
			return err
		}
	}

	return nil
}
