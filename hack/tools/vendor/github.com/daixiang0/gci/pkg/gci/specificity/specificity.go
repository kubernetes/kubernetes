package specificity

type specificityClass int

const (
	MisMatchClass        = 0
	DefaultClass         = 10
	StandardPackageClass = 20
	MatchClass           = 30
)

// MatchSpecificity is used to determine which section matches an import best
type MatchSpecificity interface {
	IsMoreSpecific(than MatchSpecificity) bool
	Equal(to MatchSpecificity) bool
	class() specificityClass
}

//Unbound methods that are required until interface methods are supported

func isMoreSpecific(this, than MatchSpecificity) bool {
	return this.class() > than.class()
}

func equalSpecificity(base, to MatchSpecificity) bool {
	// m.class() == to.class() would not work for Match
	return !base.IsMoreSpecific(to) && !to.IsMoreSpecific(base)
}
