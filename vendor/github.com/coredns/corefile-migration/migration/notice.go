package migration

import "fmt"

// Notice is a migration warning
type Notice struct {
	Plugin     string
	Option     string
	Severity   string // 'deprecated', 'removed', or 'unsupported'
	ReplacedBy string
	Additional string
	Version    string
}

func (n *Notice) ToString() string {
	s := ""
	if n.Option == "" {
		s += fmt.Sprintf(`Plugin "%v" `, n.Plugin)
	} else {
		s += fmt.Sprintf(`Option "%v" in plugin "%v" `, n.Option, n.Plugin)
	}
	if n.Severity == SevUnsupported {
		s += "is unsupported by this migration tool in " + n.Version + "."
	} else if n.Severity == SevNewDefault {
		s += "is added as a default in " + n.Version + "."
	} else {
		s += "is " + n.Severity + " in " + n.Version + "."
	}
	if n.ReplacedBy != "" {
		s += fmt.Sprintf(` It is replaced by "%v".`, n.ReplacedBy)
	}
	if n.Additional != "" {
		s += " " + n.Additional
	}
	return s
}

const (
	// The following statuses are used to indicate the state of support/deprecation in a given release.
	SevDeprecated  = "deprecated"  // deprecated, but still completely functional
	SevIgnored     = "ignored"     // if included in the corefile, it will be ignored by CoreDNS
	SevRemoved     = "removed"     // completely removed from CoreDNS, and would cause CoreDNS to exit if present in the Corefile
	SevNewDefault  = "newdefault"  // added to the default corefile.  CoreDNS may not function properly if it is not present in the corefile.
	SevUnsupported = "unsupported" // the plugin/option is not supported by the migration tool

	// The following statuses are used for selecting/filtering notifications
	SevAll = "all" // show all statuses
)
