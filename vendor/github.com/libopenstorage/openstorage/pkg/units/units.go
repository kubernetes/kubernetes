package units

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

const (
	_ = iota
	// KiB 1024 bytes
	KiB = 1 << (10 * iota)
	// MiB 1024 KiB
	MiB
	// GiB 1024 MiB
	GiB
	// TiB 1024 GiB
	TiB
	// PiB 1024 TiB
	PiB
)

const (
	// KB 1000 bytes
	KB = 1000
	// MB 1000 KB
	MB = KB * 1000
	// GB 1000 MB
	GB = MB * 1000
	// TB 1000 GB
	TB = GB * 1000
	// PB 1000 TB
	PB = TB * 1000
)

var (
	unitMap = map[string]int64{
		"B": 1,
		"b": 1,

		"KB": KiB,
		"kb": KiB,
		"MB": MiB,
		"mb": MiB,
		"GB": GiB,
		"gb": GiB,
		"TB": TiB,
		"tb": TiB,
		"PB": PiB,
		"pb": PiB,

		"K": KiB,
		"k": KiB,
		"M": MiB,
		"m": MiB,
		"G": GiB,
		"g": GiB,
		"T": TiB,
		"t": TiB,
		"P": PiB,
		"p": PiB,

		"KiB": KiB,
		"MiB": MiB,
		"GiB": GiB,
		"TiB": TiB,
		"PiB": PiB,

		"Mi": MiB,
		"Gi": GiB,
		"Ti": TiB,
		"Pi": PiB,
	}
)

var unitPattern = regexp.MustCompile(
	"([0-9]+)(.[0-9]+)*\\s*(B|b|K|k|M|m|G|g|T|t|P|p|KB|kb|KiB|MB|mb|MiB|Mi|GB|gb|GiB|Gi|TB|tb|TiB|Ti|PB|pb|PiB|Pi|)")
var BadUnit = errors.New("Bad unit")

func String(b uint64) string {
	if b > PiB {
		return fmt.Sprintf("%.2f PiB", float64(b)/float64(PiB))
	}
	if b > TiB {
		return fmt.Sprintf("%.2f TiB", float64(b)/float64(TiB))
	}
	if b > GiB {
		return fmt.Sprintf("%.1f TiB", float64(b)/float64(GiB))
	}
	if b > MiB {
		return fmt.Sprintf("%v MiB", b/MiB)
	}
	if b > KiB {
		return fmt.Sprintf("%v KiB", b/KiB)
	}
	return fmt.Sprintf("%v bytes", b)
}

func Parse(bUnit string) (int64, error) {
	ustring := strings.TrimSpace(bUnit)
	unitPattern.Longest()
	if !unitPattern.MatchString(ustring) {
		return -1, fmt.Errorf("Unit parse error: %s", bUnit)
	}
	matches := unitPattern.FindStringSubmatch(ustring)

	if len(matches) == 0 || len(matches) > 4 {
		return -1, fmt.Errorf(
			"Unit parse error: invalid count of fields (%v)",
			len(matches))
	}
	if len(matches) == 1 {
		return strconv.ParseInt(ustring, 10, 64)
	}
	shift := 0
	if len(matches) == 4 {
		shift = 1
	}
	if len(matches) == 2 {
		return -1, fmt.Errorf("Unit parse error: invalid fields %v",
			matches)
	}
	if ustring != matches[0] {
		return -1, fmt.Errorf("Unit parse error: invalid fields %v",
			matches)
	}
	multiplier, ok := unitMap[matches[2+shift]]
	if !ok {
		multiplier = unitMap["G"]
	}
	base, err := strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return -1, fmt.Errorf("Invalid number")
	}

	return base * multiplier, nil
}
