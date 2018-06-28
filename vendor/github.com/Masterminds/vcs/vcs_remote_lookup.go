package vcs

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

type vcsInfo struct {
	host     string
	pattern  string
	vcs      Type
	addCheck func(m map[string]string, u *url.URL) (Type, error)
	regex    *regexp.Regexp
}

// scpSyntaxRe matches the SCP-like addresses used by Git to access
// repositories by SSH.
var scpSyntaxRe = regexp.MustCompile(`^([a-zA-Z0-9_]+)@([a-zA-Z0-9._-]+):(.*)$`)

var vcsList = []*vcsInfo{
	{
		host:    "github.com",
		vcs:     Git,
		pattern: `^(github\.com[/|:][A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`,
	},
	{
		host:     "bitbucket.org",
		pattern:  `^(bitbucket\.org/(?P<name>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		addCheck: checkBitbucket,
	},
	{
		host:    "launchpad.net",
		pattern: `^(launchpad\.net/(([A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)?|~[A-Za-z0-9_.\-]+/(\+junk|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		vcs:     Bzr,
	},
	{
		host:    "git.launchpad.net",
		vcs:     Git,
		pattern: `^(git\.launchpad\.net/(([A-Za-z0-9_.\-]+)|~[A-Za-z0-9_.\-]+/(\+git|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))$`,
	},
	{
		host:    "hub.jazz.net",
		vcs:     Git,
		pattern: `^(hub\.jazz\.net/git/[a-z0-9]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`,
	},
	{
		host:    "go.googlesource.com",
		vcs:     Git,
		pattern: `^(go\.googlesource\.com/[A-Za-z0-9_.\-]+/?)$`,
	},
	{
		host:    "git.openstack.org",
		vcs:     Git,
		pattern: `^(git\.openstack\.org/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)$`,
	},
	// If none of the previous detect the type they will fall to this looking for the type in a generic sense
	// by the extension to the path.
	{
		addCheck: checkURL,
		pattern:  `\.(?P<type>git|hg|svn|bzr)$`,
	},
}

func init() {
	// Precompile the regular expressions used to check VCS locations.
	for _, v := range vcsList {
		v.regex = regexp.MustCompile(v.pattern)
	}
}

// This function is really a hack around Go redirects rather than around
// something VCS related. Should this be moved to the glide project or a
// helper function?
func detectVcsFromRemote(vcsURL string) (Type, string, error) {
	t, e := detectVcsFromURL(vcsURL)
	if e == nil {
		return t, vcsURL, nil
	} else if e != ErrCannotDetectVCS {
		return NoVCS, "", e
	}

	// Pages like https://golang.org/x/net provide an html document with
	// meta tags containing a location to work with. The go tool uses
	// a meta tag with the name go-import which is what we use here.
	// godoc.org also has one call go-source that we do not need to use.
	// The value of go-import is in the form "prefix vcs repo". The prefix
	// should match the vcsURL and the repo is a location that can be
	// checked out. Note, to get the html document you you need to add
	// ?go-get=1 to the url.
	u, err := url.Parse(vcsURL)
	if err != nil {
		return NoVCS, "", err
	}
	if u.RawQuery == "" {
		u.RawQuery = "go-get=1"
	} else {
		u.RawQuery = u.RawQuery + "+go-get=1"
	}
	checkURL := u.String()
	resp, err := http.Get(checkURL)
	if err != nil {
		return NoVCS, "", ErrCannotDetectVCS
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp.StatusCode == 404 {
			return NoVCS, "", NewRemoteError(fmt.Sprintf("%s Not Found", vcsURL), nil, "")
		} else if resp.StatusCode == 401 || resp.StatusCode == 403 {
			return NoVCS, "", NewRemoteError(fmt.Sprintf("%s Access Denied", vcsURL), nil, "")
		}
		return NoVCS, "", ErrCannotDetectVCS
	}

	t, nu, err := parseImportFromBody(u, resp.Body)
	if err != nil {
		// TODO(mattfarina): Log the parsing error
		return NoVCS, "", ErrCannotDetectVCS
	} else if t == "" || nu == "" {
		return NoVCS, "", ErrCannotDetectVCS
	}

	return t, nu, nil
}

// From a remote vcs url attempt to detect the VCS.
func detectVcsFromURL(vcsURL string) (Type, error) {

	var u *url.URL
	var err error

	if m := scpSyntaxRe.FindStringSubmatch(vcsURL); m != nil {
		// Match SCP-like syntax and convert it to a URL.
		// Eg, "git@github.com:user/repo" becomes
		// "ssh://git@github.com/user/repo".
		u = &url.URL{
			Scheme: "ssh",
			User:   url.User(m[1]),
			Host:   m[2],
			Path:   "/" + m[3],
		}
	} else {
		u, err = url.Parse(vcsURL)
		if err != nil {
			return "", err
		}
	}

	// Detect file schemes
	if u.Scheme == "file" {
		return DetectVcsFromFS(u.Path)
	}

	if u.Host == "" {
		return "", ErrCannotDetectVCS
	}

	// Try to detect from the scheme
	switch u.Scheme {
	case "git+ssh":
		return Git, nil
	case "git":
		return Git, nil
	case "bzr+ssh":
		return Bzr, nil
	case "svn+ssh":
		return Svn, nil
	}

	// Try to detect from known hosts, such as Github
	for _, v := range vcsList {
		if v.host != "" && v.host != u.Host {
			continue
		}

		// Make sure the pattern matches for an actual repo location. For example,
		// we should fail if the VCS listed is github.com/masterminds as that's
		// not actually a repo.
		uCheck := u.Host + u.Path
		m := v.regex.FindStringSubmatch(uCheck)
		if m == nil {
			if v.host != "" {
				return "", ErrCannotDetectVCS
			}

			continue
		}

		// If we are here the host matches. If the host has a singular
		// VCS type, such as Github, we can return the type right away.
		if v.vcs != "" {
			return v.vcs, nil
		}

		// Run additional checks to determine try and determine the repo
		// for the matched service.
		info := make(map[string]string)
		for i, name := range v.regex.SubexpNames() {
			if name != "" {
				info[name] = m[i]
			}
		}
		t, err := v.addCheck(info, u)
		if err != nil {
			switch err.(type) {
			case *RemoteError:
				return "", err
			}
			return "", ErrCannotDetectVCS
		}

		return t, nil
	}

	// Attempt to ascertain from the username passed in.
	if u.User != nil {
		un := u.User.Username()
		if un == "git" {
			return Git, nil
		} else if un == "hg" {
			return Hg, nil
		}
	}

	// Unable to determine the vcs from the url.
	return "", ErrCannotDetectVCS
}

// Figure out the type for Bitbucket by the passed in information
// or via the public API.
func checkBitbucket(i map[string]string, ul *url.URL) (Type, error) {

	// Fast path for ssh urls where we may not even be able to
	// anonymously get details from the API.
	if ul.User != nil {
		un := ul.User.Username()
		if un == "git" {
			return Git, nil
		} else if un == "hg" {
			return Hg, nil
		}
	}

	// The part of the response we care about.
	var response struct {
		SCM Type `json:"scm"`
	}

	u := expand(i, "https://api.bitbucket.org/1.0/repositories/{name}")
	data, err := get(u)
	if err != nil {
		return "", err
	}

	if err := json.Unmarshal(data, &response); err != nil {
		return "", fmt.Errorf("Decoding error %s: %v", u, err)
	}

	return response.SCM, nil

}

// Expect a type key on i with the exact type detected from the regex.
func checkURL(i map[string]string, u *url.URL) (Type, error) {
	return Type(i["type"]), nil
}

func get(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		if resp.StatusCode == 404 {
			return nil, NewRemoteError("Not Found", err, resp.Status)
		} else if resp.StatusCode == 401 || resp.StatusCode == 403 {
			return nil, NewRemoteError("Access Denied", err, resp.Status)
		}
		return nil, fmt.Errorf("%s: %s", url, resp.Status)
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", url, err)
	}
	return b, nil
}

func expand(match map[string]string, s string) string {
	for k, v := range match {
		s = strings.Replace(s, "{"+k+"}", v, -1)
	}
	return s
}

func parseImportFromBody(ur *url.URL, r io.ReadCloser) (tp Type, u string, err error) {
	d := xml.NewDecoder(r)
	d.CharsetReader = charsetReader
	d.Strict = false
	var t xml.Token
	for {
		t, err = d.Token()
		if err != nil {
			if err == io.EOF {
				// When the end is reached it could not detect a VCS if it
				// got here.
				err = ErrCannotDetectVCS
			}
			return
		}
		if e, ok := t.(xml.StartElement); ok && strings.EqualFold(e.Name.Local, "body") {
			return
		}
		if e, ok := t.(xml.EndElement); ok && strings.EqualFold(e.Name.Local, "head") {
			return
		}
		e, ok := t.(xml.StartElement)
		if !ok || !strings.EqualFold(e.Name.Local, "meta") {
			continue
		}
		if attrValue(e.Attr, "name") != "go-import" {
			continue
		}
		if f := strings.Fields(attrValue(e.Attr, "content")); len(f) == 3 {
			// If the prefix supplied by the remote system isn't a prefix to the
			// url we're fetching continue to look for other imports.
			// This will work for exact matches and prefixes. For example,
			// golang.org/x/net as a prefix will match for golang.org/x/net and
			// golang.org/x/net/context.
			vcsURL := ur.Host + ur.Path
			if !strings.HasPrefix(vcsURL, f[0]) {
				continue
			} else {
				switch Type(f[1]) {
				case Git:
					tp = Git
				case Svn:
					tp = Svn
				case Bzr:
					tp = Bzr
				case Hg:
					tp = Hg
				}

				u = f[2]
				return
			}
		}
	}
}

func charsetReader(charset string, input io.Reader) (io.Reader, error) {
	switch strings.ToLower(charset) {
	case "ascii":
		return input, nil
	default:
		return nil, fmt.Errorf("can't decode XML document using charset %q", charset)
	}
}

func attrValue(attrs []xml.Attr, name string) string {
	for _, a := range attrs {
		if strings.EqualFold(a.Name.Local, name) {
			return a.Value
		}
	}
	return ""
}
