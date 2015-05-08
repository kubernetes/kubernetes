package types

type Port struct {
	Name            ACName `json:"name"`
	Protocol        string `json:"protocol"`
	Port            uint   `json:"port"`
	SocketActivated bool   `json:"socketActivated"`
}

type ExposedPort struct {
	Name     ACName `json:"name"`
	HostPort uint   `json:"hostPort"`
}
