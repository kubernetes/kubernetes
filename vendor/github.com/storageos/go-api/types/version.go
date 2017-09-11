package types

// VersionInfo describes version and runtime info.
type VersionInfo struct {
	Name          string `json:"name"`
	BuildDate     string `json:"buildDate"`
	Revision      string `json:"revision"`
	Version       string `json:"version"`
	APIVersion    string `json:"apiVersion"`
	GoVersion     string `json:"goVersion"`
	OS            string `json:"os"`
	Arch          string `json:"arch"`
	KernelVersion string `json:"kernelVersion"`
	Experimental  bool   `json:"experimental"`
}

type VersionResponse struct {
	Client *VersionInfo
	Server *VersionInfo
}

// ServerOK returns true when the client could connect to the docker server
// and parse the information received. It returns false otherwise.
func (v VersionResponse) ServerOK() bool {
	return v.Server != nil
}
