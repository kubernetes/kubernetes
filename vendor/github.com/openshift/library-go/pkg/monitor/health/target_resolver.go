package health

// Listener is an interface to use to notify interested parties of a change.
type Listener interface {
	// Enqueue should be called when an input may have changed
	Enqueue()
}

// Notifier is a way to add listeners
type Notifier interface {
	// AddListener is adds a listener to be notified of potential input changes
	AddListener(listener Listener)
}

// TargetProviders is an interface to use to get a list of targets to monitor
type TargetProvider interface {
	// CurrentTargetsList returns a precomputed list of targets
	CurrentTargetsList() []string
}

// StaticTargetProvider implements TargetProvider and provides a static list of targets
type StaticTargetProvider []string

var _ TargetProvider = StaticTargetProvider{}

func (sp StaticTargetProvider) CurrentTargetsList() []string {
	return sp
}
