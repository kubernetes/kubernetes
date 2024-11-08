package extensiontests

func (results ExtensionTestResults) Walk(walkFn func(*ExtensionTestResult)) {
	for i := range results {
		walkFn(results[i])
	}
}

// AddDetails adds additional information to an ExtensionTestResult. Value must marshal to JSON.
func (result *ExtensionTestResult) AddDetails(name string, value interface{}) {
	result.Details = append(result.Details, Details{Name: name, Value: value})
}
