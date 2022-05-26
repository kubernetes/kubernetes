package gosec

// ReportInfo this is report information
type ReportInfo struct {
	Errors       map[string][]Error `json:"Golang errors"`
	Issues       []*Issue
	Stats        *Metrics
	GosecVersion string
}

// NewReportInfo instantiate a ReportInfo
func NewReportInfo(issues []*Issue, metrics *Metrics, errors map[string][]Error) *ReportInfo {
	return &ReportInfo{
		Errors: errors,
		Issues: issues,
		Stats:  metrics,
	}
}

// WithVersion defines the version of gosec used to generate the report
func (r *ReportInfo) WithVersion(version string) *ReportInfo {
	r.GosecVersion = version
	return r
}
