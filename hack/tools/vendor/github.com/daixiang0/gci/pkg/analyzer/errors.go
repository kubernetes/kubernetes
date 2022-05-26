package analyzer

import "fmt"

type InvalidNumberOfFilesInAnalysis struct {
	expectedNumFiles, foundNumFiles int
}

func (i InvalidNumberOfFilesInAnalysis) Error() string {
	return fmt.Sprintf("Expected %d files in Analyzer input, Found %d", i.expectedNumFiles, i.foundNumFiles)
}

func (i InvalidNumberOfFilesInAnalysis) Is(err error) bool {
	_, ok := err.(InvalidNumberOfFilesInAnalysis)
	return ok
}
