package config

// GinkgoConfigType has been deprecated and its equivalent now lives in
// the types package.  You can no longer access Ginkgo configuration from the config
// package.  Instead use the DSL's GinkgoConfiguration() function to get copies of the
// current configuration
//
// GinkgoConfigType is still here so custom V1 reporters do not result in a compilation error
// It will be removed in a future minor release of Ginkgo
type GinkgoConfigType = DeprecatedGinkgoConfigType
type DeprecatedGinkgoConfigType struct {
	RandomSeed         int64
	RandomizeAllSpecs  bool
	RegexScansFilePath bool
	FocusStrings       []string
	SkipStrings        []string
	SkipMeasurements   bool
	FailOnPending      bool
	FailFast           bool
	FlakeAttempts      int
	EmitSpecProgress   bool
	DryRun             bool
	DebugParallel      bool

	ParallelNode  int
	ParallelTotal int
	SyncHost      string
	StreamHost    string
}

// DefaultReporterConfigType has been deprecated and its equivalent now lives in
// the types package.  You can no longer access Ginkgo configuration from the config
// package.  Instead use the DSL's GinkgoConfiguration() function to get copies of the
// current configuration
//
// DefaultReporterConfigType is still here so custom V1 reporters do not result in a compilation error
// It will be removed in a future minor release of Ginkgo
type DefaultReporterConfigType = DeprecatedDefaultReporterConfigType
type DeprecatedDefaultReporterConfigType struct {
	NoColor           bool
	SlowSpecThreshold float64
	NoisyPendings     bool
	NoisySkippings    bool
	Succinct          bool
	Verbose           bool
	FullTrace         bool
	ReportPassed      bool
	ReportFile        string
}

// Sadly there is no way to graefully deprecate access to these global config variables.
// Users who need access to Ginkgo's configuration should use the DLS's GinkgoConfiguraiton() method
// These new unwieldy type names exist to give users a hint when they try to compile and the compilation fails
type GinkgoConfigIsNoLongerAccessibleFromTheConfigPackageUseTheDLSsGinkgoConfigurationFunctionInstead struct{}

// Sadly there is no way to graefully deprecate access to these global config variables.
// Users who need access to Ginkgo's configuration should use the DLS's GinkgoConfiguraiton() method
// These new unwieldy type names exist to give users a hint when they try to compile and the compilation fails
var GinkgoConfig = GinkgoConfigIsNoLongerAccessibleFromTheConfigPackageUseTheDLSsGinkgoConfigurationFunctionInstead{}

// Sadly there is no way to graefully deprecate access to these global config variables.
// Users who need access to Ginkgo's configuration should use the DLS's GinkgoConfiguraiton() method
// These new unwieldy type names exist to give users a hint when they try to compile and the compilation fails
type DefaultReporterConfigIsNoLongerAccessibleFromTheConfigPackageUseTheDLSsGinkgoConfigurationFunctionInstead struct{}

// Sadly there is no way to graefully deprecate access to these global config variables.
// Users who need access to Ginkgo's configuration should use the DLS's GinkgoConfiguraiton() method
// These new unwieldy type names exist to give users a hint when they try to compile and the compilation fails
var DefaultReporterConfig = DefaultReporterConfigIsNoLongerAccessibleFromTheConfigPackageUseTheDLSsGinkgoConfigurationFunctionInstead{}
