package lint

// FormatterMetadata configuration of a formatter
type FormatterMetadata struct {
	Name        string
	Description string
	Sample      string
}

// Formatter defines an interface for failure formatters
type Formatter interface {
	Format(<-chan Failure, Config) (string, error)
	Name() string
}
