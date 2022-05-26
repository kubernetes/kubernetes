package exitcodes

const (
	Success = iota
	IssuesFound
	WarningInTest
	Failure
	Timeout
	NoGoFiles
	NoConfigFileDetected
	ErrorWasLogged
)

type ExitError struct {
	Message string
	Code    int
}

func (e ExitError) Error() string {
	return e.Message
}

var (
	ErrNoGoFiles = &ExitError{
		Message: "no go files to analyze",
		Code:    NoGoFiles,
	}
	ErrFailure = &ExitError{
		Message: "failed to analyze",
		Code:    Failure,
	}
)
