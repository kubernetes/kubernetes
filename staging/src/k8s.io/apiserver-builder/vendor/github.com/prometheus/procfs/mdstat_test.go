package procfs

import (
	"testing"
)

func TestMDStat(t *testing.T) {
	fs := FS("fixtures")
	mdStates, err := fs.ParseMDStat()
	if err != nil {
		t.Fatalf("parsing of reference-file failed entirely: %s", err)
	}

	refs := map[string]MDStat{
		"md3":   MDStat{"md3", "active", 8, 8, 5853468288, 5853468288},
		"md127": MDStat{"md127", "active", 2, 2, 312319552, 312319552},
		"md0":   MDStat{"md0", "active", 2, 2, 248896, 248896},
		"md4":   MDStat{"md4", "inactive", 2, 2, 4883648, 4883648},
		"md6":   MDStat{"md6", "active", 1, 2, 195310144, 16775552},
		"md8":   MDStat{"md8", "active", 2, 2, 195310144, 16775552},
		"md7":   MDStat{"md7", "active", 3, 4, 7813735424, 7813735424},
	}

	for _, md := range mdStates {
		if md != refs[md.Name] {
			t.Errorf("failed parsing md-device %s correctly: want %v, got %v", md.Name, refs[md.Name], md)
		}
	}

	if want, have := len(refs), len(mdStates); want != have {
		t.Errorf("want %d parsed md-devices, have %d", want, have)
	}
}
