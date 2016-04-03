package registry

type SearchResult struct {
	StarCount   int    `json:"star_count"`
	IsOfficial  bool   `json:"is_official"`
	Name        string `json:"name"`
	IsTrusted   bool   `json:"is_trusted"`
	IsAutomated bool   `json:"is_automated"`
	Description string `json:"description"`
}

type SearchResults struct {
	Query      string         `json:"query"`
	NumResults int            `json:"num_results"`
	Results    []SearchResult `json:"results"`
}

type RepositoryData struct {
	ImgList   map[string]*ImgData
	Endpoints []string
	Tokens    []string
}

type ImgData struct {
	ID              string `json:"id"`
	Checksum        string `json:"checksum,omitempty"`
	ChecksumPayload string `json:"-"`
	Tag             string `json:",omitempty"`
}

type RegistryInfo struct {
	Version    string `json:"version"`
	Standalone bool   `json:"standalone"`
}

type APIVersion int

func (av APIVersion) String() string {
	return apiVersions[av]
}

var apiVersions = map[APIVersion]string{
	1: "v1",
	2: "v2",
}

// API Version identifiers.
const (
	APIVersionUnknown = iota
	APIVersion1
	APIVersion2
)

// RepositoryInfo Examples:
// {
//   "Index" : {
//     "Name" : "docker.io",
//     "Mirrors" : ["https://registry-2.docker.io/v1/", "https://registry-3.docker.io/v1/"],
//     "Secure" : true,
//     "Official" : true,
//   },
//   "RemoteName" : "library/debian",
//   "LocalName" : "debian",
//   "CanonicalName" : "docker.io/debian"
//   "Official" : true,
// }

// {
//   "Index" : {
//     "Name" : "127.0.0.1:5000",
//     "Mirrors" : [],
//     "Secure" : false,
//     "Official" : false,
//   },
//   "RemoteName" : "user/repo",
//   "LocalName" : "127.0.0.1:5000/user/repo",
//   "CanonicalName" : "127.0.0.1:5000/user/repo",
//   "Official" : false,
// }
type IndexInfo struct {
	Name     string
	Mirrors  []string
	Secure   bool
	Official bool
}

type RepositoryInfo struct {
	Index         *IndexInfo
	RemoteName    string
	LocalName     string
	CanonicalName string
	Official      bool
}
