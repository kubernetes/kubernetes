package lint

const (
	StateInitializing = 0
	StateGraph        = 1
	StateProcessing   = 2
	StateCumulative   = 3
)

type Stats struct {
	State uint32

	InitialPackages          uint32
	TotalPackages            uint32
	ProcessedPackages        uint32
	ProcessedInitialPackages uint32
	Problems                 uint32
	ActiveWorkers            uint32
	TotalWorkers             uint32
}
