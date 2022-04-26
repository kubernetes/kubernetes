package plumbing

// Revision represents a git revision
// to get more details about git revisions
// please check git manual page :
// https://www.kernel.org/pub/software/scm/git/docs/gitrevisions.html
type Revision string

func (r Revision) String() string {
	return string(r)
}
