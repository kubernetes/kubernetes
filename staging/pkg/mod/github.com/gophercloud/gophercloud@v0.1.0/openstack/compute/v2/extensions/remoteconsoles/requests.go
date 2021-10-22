package remoteconsoles

import (
	"github.com/gophercloud/gophercloud"
)

// ConsoleProtocol represents valid remote console protocol.
// It can be used to create a remote console with one of the pre-defined protocol.
type ConsoleProtocol string

const (
	// ConsoleProtocolVNC represents the VNC console protocol.
	ConsoleProtocolVNC ConsoleProtocol = "vnc"

	// ConsoleProtocolSPICE represents the SPICE console protocol.
	ConsoleProtocolSPICE ConsoleProtocol = "spice"

	// ConsoleProtocolRDP represents the RDP console protocol.
	ConsoleProtocolRDP ConsoleProtocol = "rdp"

	// ConsoleProtocolSerial represents the Serial console protocol.
	ConsoleProtocolSerial ConsoleProtocol = "serial"

	// ConsoleProtocolMKS represents the MKS console protocol.
	ConsoleProtocolMKS ConsoleProtocol = "mks"
)

// ConsoleType represents valid remote console type.
// It can be used to create a remote console with one of the pre-defined type.
type ConsoleType string

const (
	// ConsoleTypeNoVNC represents the VNC console type.
	ConsoleTypeNoVNC ConsoleType = "novnc"

	// ConsoleTypeXVPVNC represents the XVP VNC console type.
	ConsoleTypeXVPVNC ConsoleType = "xvpvnc"

	// ConsoleTypeRDPHTML5 represents the RDP HTML5 console type.
	ConsoleTypeRDPHTML5 ConsoleType = "rdp-html5"

	// ConsoleTypeSPICEHTML5 represents the SPICE HTML5 console type.
	ConsoleTypeSPICEHTML5 ConsoleType = "spice-html5"

	// ConsoleTypeSerial represents the Serial console type.
	ConsoleTypeSerial ConsoleType = "serial"

	// ConsoleTypeWebMKS represents the Web MKS console type.
	ConsoleTypeWebMKS ConsoleType = "webmks"
)

// CreateOptsBuilder allows to add additional parameters to the Create request.
type CreateOptsBuilder interface {
	ToRemoteConsoleCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters to the Create request.
type CreateOpts struct {
	// Protocol specifies the protocol of a new remote console.
	Protocol ConsoleProtocol `json:"protocol" required:"true"`

	// Type specifies the type of a new remote console.
	Type ConsoleType `json:"type" required:"true"`
}

// ToRemoteConsoleCreateMap builds a request body from the CreateOpts.
func (opts CreateOpts) ToRemoteConsoleCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "remote_console")
}

// Create requests the creation of a new remote console on the specified server.
func Create(client *gophercloud.ServiceClient, serverID string, opts CreateOptsBuilder) (r CreateResult) {
	reqBody, err := opts.ToRemoteConsoleCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client, serverID), reqBody, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
