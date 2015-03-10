package mount

import (
	"strings"
)

// Parse fstab type mount options into mount() flags
// and device specific data
func parseOptions(options string) (int, string) {
	var (
		flag int
		data []string
	)

	flags := map[string]struct {
		clear bool
		flag  int
	}{
		"defaults":      {false, 0},
		"ro":            {false, RDONLY},
		"rw":            {true, RDONLY},
		"suid":          {true, NOSUID},
		"nosuid":        {false, NOSUID},
		"dev":           {true, NODEV},
		"nodev":         {false, NODEV},
		"exec":          {true, NOEXEC},
		"noexec":        {false, NOEXEC},
		"sync":          {false, SYNCHRONOUS},
		"async":         {true, SYNCHRONOUS},
		"dirsync":       {false, DIRSYNC},
		"remount":       {false, REMOUNT},
		"mand":          {false, MANDLOCK},
		"nomand":        {true, MANDLOCK},
		"atime":         {true, NOATIME},
		"noatime":       {false, NOATIME},
		"diratime":      {true, NODIRATIME},
		"nodiratime":    {false, NODIRATIME},
		"bind":          {false, BIND},
		"rbind":         {false, RBIND},
		"unbindable":    {false, UNBINDABLE},
		"runbindable":   {false, RUNBINDABLE},
		"private":       {false, PRIVATE},
		"rprivate":      {false, RPRIVATE},
		"shared":        {false, SHARED},
		"rshared":       {false, RSHARED},
		"slave":         {false, SLAVE},
		"rslave":        {false, RSLAVE},
		"relatime":      {false, RELATIME},
		"norelatime":    {true, RELATIME},
		"strictatime":   {false, STRICTATIME},
		"nostrictatime": {true, STRICTATIME},
	}

	for _, o := range strings.Split(options, ",") {
		// If the option does not exist in the flags table or the flag
		// is not supported on the platform,
		// then it is a data value for a specific fs type
		if f, exists := flags[o]; exists && f.flag != 0 {
			if f.clear {
				flag &= ^f.flag
			} else {
				flag |= f.flag
			}
		} else {
			data = append(data, o)
		}
	}
	return flag, strings.Join(data, ",")
}
