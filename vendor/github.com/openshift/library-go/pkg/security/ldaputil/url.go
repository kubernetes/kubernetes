package ldaputil

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/go-ldap/ldap/v3"
)

// Scheme is a valid ldap scheme
type Scheme string

const (
	SchemeLDAP  Scheme = "ldap"
	SchemeLDAPS Scheme = "ldaps"
)

// Scope is a valid LDAP search scope
type Scope int

const (
	ScopeWholeSubtree Scope = ldap.ScopeWholeSubtree
	ScopeSingleLevel  Scope = ldap.ScopeSingleLevel
	ScopeBaseObject   Scope = ldap.ScopeBaseObject
)

// DerefAliases is a valid LDAP alias dereference parameter
type DerefAliases int

const (
	DerefAliasesNever     = ldap.NeverDerefAliases
	DerefAliasesSearching = ldap.DerefInSearching
	DerefAliasesFinding   = ldap.DerefFindingBaseObj
	DerefAliasesAlways    = ldap.DerefAlways
)

const (
	defaultLDAPPort  = "389"
	defaultLDAPSPort = "636"

	defaultHost           = "localhost"
	defaultQueryAttribute = "uid"
	defaultFilter         = "(objectClass=*)"

	scopeWholeSubtreeString = "sub"
	scopeSingleLevelString  = "one"
	scopeBaseObjectString   = "base"

	criticalExtensionPrefix = "!"
)

// LDAPURL holds a parsed RFC 2255 URL
type LDAPURL struct {
	// Scheme is ldap or ldaps
	Scheme Scheme
	// Host is the host:port of the LDAP server
	Host string
	// The DN of the branch of the directory where all searches should start from
	BaseDN string
	// The attribute to search for
	QueryAttribute string
	// The scope of the search. Can be ldap.ScopeWholeSubtree, ldap.ScopeSingleLevel, or ldap.ScopeBaseObject
	Scope Scope
	// A valid LDAP search filter (e.g. "(objectClass=*)")
	Filter string
}

// ParseURL parsed the given ldapURL as an RFC 2255 URL
// The syntax of the URL is ldap://host:port/basedn?attribute?scope?filter
func ParseURL(ldapURL string) (LDAPURL, error) {
	// Must be a valid URL to start
	parsedURL, err := url.Parse(ldapURL)
	if err != nil {
		return LDAPURL{}, err
	}

	opts := LDAPURL{}

	determinedScheme, err := DetermineLDAPScheme(parsedURL.Scheme)
	if err != nil {
		return LDAPURL{}, err
	}
	opts.Scheme = determinedScheme

	determinedHost, err := DetermineLDAPHost(parsedURL.Host, opts.Scheme)
	if err != nil {
		return LDAPURL{}, err
	}
	opts.Host = determinedHost

	// Set base dn (default to "")
	// url.Parse() already percent-decodes the path
	opts.BaseDN = strings.TrimLeft(parsedURL.Path, "/")

	attributes, scope, filter, extensions, err := SplitLDAPQuery(parsedURL.RawQuery)
	if err != nil {
		return LDAPURL{}, err
	}

	// Attributes contains comma-separated attributes
	// Set query attribute to first attribute
	// Default to uid to match mod_auth_ldap
	opts.QueryAttribute = strings.Split(attributes, ",")[0]
	if len(opts.QueryAttribute) == 0 {
		opts.QueryAttribute = defaultQueryAttribute
	}

	determinedScope, err := DetermineLDAPScope(scope)
	if err != nil {
		return LDAPURL{}, err
	}
	opts.Scope = determinedScope

	determinedFilter, err := DetermineLDAPFilter(filter)
	if err != nil {
		return LDAPURL{}, err
	}
	opts.Filter = determinedFilter

	// Extensions are in "name=value,name2=value2" form
	// Critical extensions are prefixed with a !
	// Optional extensions are ignored, per RFC
	// Fail if there are any critical extensions, since we don't support any
	if len(extensions) > 0 {
		for _, extension := range strings.Split(extensions, ",") {
			exttype := strings.SplitN(extension, "=", 2)[0]
			if strings.HasPrefix(exttype, criticalExtensionPrefix) {
				return LDAPURL{}, fmt.Errorf("unsupported critical extension %s", extension)
			}
		}
	}

	return opts, nil

}

// DetermineLDAPScheme determines the LDAP connection scheme. Scheme is one of "ldap" or "ldaps"
// Default to "ldap"
func DetermineLDAPScheme(scheme string) (Scheme, error) {
	switch Scheme(scheme) {
	case SchemeLDAP, SchemeLDAPS:
		return Scheme(scheme), nil
	default:
		return "", fmt.Errorf("invalid scheme %q", scheme)
	}
}

// DetermineLDAPHost determines the host and port for the LDAP connection.
// The default host is localhost; the default port for scheme "ldap" is 389, for "ldaps" is 686
func DetermineLDAPHost(hostport string, scheme Scheme) (string, error) {
	if len(hostport) == 0 {
		hostport = defaultHost
	}
	// add port if missing
	if _, _, err := net.SplitHostPort(hostport); err != nil {
		switch scheme {
		case SchemeLDAPS:
			return net.JoinHostPort(hostport, defaultLDAPSPort), nil
		case SchemeLDAP:
			return net.JoinHostPort(hostport, defaultLDAPPort), nil
		default:
			return "", fmt.Errorf("no default port for scheme %q", scheme)
		}
	}
	// nothing needed to be done
	return hostport, nil
}

// SplitLDAPQuery splits the query in the URL into the substituent parts. All sections are optional.
// Query syntax is attribute?scope?filter?extensions
func SplitLDAPQuery(query string) (attributes, scope, filter, extensions string, err error) {
	parts := strings.Split(query, "?")
	switch len(parts) {
	case 4:
		extensions = parts[3]
		fallthrough
	case 3:
		if v, err := url.QueryUnescape(parts[2]); err != nil {
			return "", "", "", "", err
		} else {
			filter = v
		}
		fallthrough
	case 2:
		if v, err := url.QueryUnescape(parts[1]); err != nil {
			return "", "", "", "", err
		} else {
			scope = v
		}
		fallthrough
	case 1:
		if v, err := url.QueryUnescape(parts[0]); err != nil {
			return "", "", "", "", err
		} else {
			attributes = v
		}
		return attributes, scope, filter, extensions, nil
	case 0:
		return
	default:
		err = fmt.Errorf("too many query options %q", query)
		return "", "", "", "", err
	}
}

// DetermineLDAPScope determines the LDAP search scope. Scope is one of "sub", "one", or "base"
// Default to "sub" to match mod_auth_ldap
func DetermineLDAPScope(scope string) (Scope, error) {
	switch scope {
	case "", scopeWholeSubtreeString:
		return ScopeWholeSubtree, nil
	case scopeSingleLevelString:
		return ScopeSingleLevel, nil
	case scopeBaseObjectString:
		return ScopeBaseObject, nil
	default:
		return -1, fmt.Errorf("invalid scope %q", scope)
	}
}

// DetermineLDAPFilter determines the LDAP search filter. Filter is a valid LDAP filter
// Default to "(objectClass=*)" per RFC
func DetermineLDAPFilter(filter string) (string, error) {
	if len(filter) == 0 {
		return defaultFilter, nil
	}
	if _, err := ldap.CompileFilter(filter); err != nil {
		return "", fmt.Errorf("invalid filter: %v", err)
	}
	return filter, nil
}

func DetermineDerefAliasesBehavior(derefAliasesString string) (DerefAliases, error) {
	mapping := map[string]DerefAliases{
		"never":  DerefAliasesNever,
		"search": DerefAliasesSearching,
		"base":   DerefAliasesFinding,
		"always": DerefAliasesAlways,
	}
	derefAliases, exists := mapping[derefAliasesString]
	if !exists {
		return -1, fmt.Errorf("not a valid LDAP alias dereferncing behavior: %s", derefAliasesString)
	}
	return derefAliases, nil
}
