package procfs

import (
	"testing"
)

func TestMDStat(t *testing.T) {
	mdStates, err := FS("fixtures").ParseMDStat()
	if err != nil {
		t.Fatalf("parsing of reference-file failed entirely: %s", err)
	}

	refs := map[string]MDStat{
		"md3":   {"md3", "active", 8, 8, 5853468288, 5853468288},
		"md127": {"md127", "active", 2, 2, 312319552, 312319552},
		"md0":   {"md0", "active", 2, 2, 248896, 248896},
		"md4":   {"md4", "inactive", 2, 2, 4883648, 4883648},
		"md6":   {"md6", "active", 1, 2, 195310144, 16775552},
		"md8":   {"md8", "active", 2, 2, 195310144, 16775552},
		"md7":   {"md7", "active", 3, 4, 7813735424, 7813735424},
	}

	if want, have := len(refs), len(mdStates); want != have {
		t.Errorf("want %d parsed md-devices, have %d", want, have)
	}
	for _, md := range mdStates {
		if want, have := refs[md.Name], md; want != have {
			t.Errorf("%s: want %v, have %v", md.Name, want, have)
		}
	}
}
