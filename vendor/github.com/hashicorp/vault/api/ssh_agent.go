package api

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/hashicorp/go-cleanhttp"
	"github.com/hashicorp/go-multierror"
	"github.com/hashicorp/go-rootcerts"
	"github.com/hashicorp/hcl"
	"github.com/hashicorp/hcl/hcl/ast"
	"github.com/mitchellh/mapstructure"
)

const (
	// SSHHelperDefaultMountPoint is the default path at which SSH backend will be
	// mounted in the Vault server.
	SSHHelperDefaultMountPoint = "ssh"

	// VerifyEchoRequest is the echo request message sent as OTP by the helper.
	VerifyEchoRequest = "verify-echo-request"

	// VerifyEchoResponse is the echo response message sent as a response to OTP
	// matching echo request.
	VerifyEchoResponse = "verify-echo-response"
)

// SSHHelper is a structure representing a vault-ssh-helper which can talk to vault server
// in order to verify the OTP entered by the user. It contains the path at which
// SSH backend is mounted at the server.
type SSHHelper struct {
	c          *Client
	MountPoint string
}

// SSHVerifyResponse is a structure representing the fields in Vault server's
// response.
type SSHVerifyResponse struct {
	// Usually empty. If the request OTP is echo request message, this will
	// be set to the corresponding echo response message.
	Message string `json:"message" structs:"message" mapstructure:"message"`

	// Username associated with the OTP
	Username string `json:"username" structs:"username" mapstructure:"username"`

	// IP associated with the OTP
	IP string `json:"ip" structs:"ip" mapstructure:"ip"`

	// Name of the role against which the OTP was issued
	RoleName string `json:"role_name" structs:"role_name" mapstructure:"role_name"`
}

// SSHHelperConfig is a structure which represents the entries from the vault-ssh-helper's configuration file.
type SSHHelperConfig struct {
	VaultAddr       string `hcl:"vault_addr"`
	SSHMountPoint   string `hcl:"ssh_mount_point"`
	CACert          string `hcl:"ca_cert"`
	CAPath          string `hcl:"ca_path"`
	AllowedCidrList string `hcl:"allowed_cidr_list"`
	AllowedRoles    string `hcl:"allowed_roles"`
	TLSSkipVerify   bool   `hcl:"tls_skip_verify"`
	TLSServerName   string `hcl:"tls_server_name"`
}

// SetTLSParameters sets the TLS parameters for this SSH agent.
func (c *SSHHelperConfig) SetTLSParameters(clientConfig *Config, certPool *x509.CertPool) {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: c.TLSSkipVerify,
		MinVersion:         tls.VersionTLS12,
		RootCAs:            certPool,
		ServerName:         c.TLSServerName,
	}

	transport := cleanhttp.DefaultTransport()
	transport.TLSClientConfig = tlsConfig
	clientConfig.HttpClient.Transport = transport
}

// Returns true if any of the following conditions are true:
//   * CA cert is configured
//   * CA path is configured
//   * configured to skip certificate verification
//   * TLS server name is configured
//
func (c *SSHHelperConfig) shouldSetTLSParameters() bool {
	return c.CACert != "" || c.CAPath != "" || c.TLSServerName != "" || c.TLSSkipVerify
}

// NewClient returns a new client for the configuration. This client will be used by the
// vault-ssh-helper to communicate with Vault server and verify the OTP entered by user.
// If the configuration supplies Vault SSL certificates, then the client will
// have TLS configured in its transport.
func (c *SSHHelperConfig) NewClient() (*Client, error) {
	// Creating a default client configuration for communicating with vault server.
	clientConfig := DefaultConfig()

	// Pointing the client to the actual address of vault server.
	clientConfig.Address = c.VaultAddr

	// Check if certificates are provided via config file.
	if c.shouldSetTLSParameters() {
		rootConfig := &rootcerts.Config{
			CAFile: c.CACert,
			CAPath: c.CAPath,
		}
		certPool, err := rootcerts.LoadCACerts(rootConfig)
		if err != nil {
			return nil, err
		}
		// Enable TLS on the HTTP client information
		c.SetTLSParameters(clientConfig, certPool)
	}

	// Creating the client object for the given configuration
	client, err := NewClient(clientConfig)
	if err != nil {
		return nil, err
	}

	return client, nil
}

// LoadSSHHelperConfig loads ssh-helper's configuration from the file and populates the corresponding
// in-memory structure.
//
// Vault address is a required parameter.
// Mount point defaults to "ssh".
func LoadSSHHelperConfig(path string) (*SSHHelperConfig, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, multierror.Prefix(err, "ssh_helper:")
	}
	return ParseSSHHelperConfig(string(contents))
}

// ParseSSHHelperConfig parses the given contents as a string for the SSHHelper
// configuration.
func ParseSSHHelperConfig(contents string) (*SSHHelperConfig, error) {
	root, err := hcl.Parse(string(contents))
	if err != nil {
		return nil, fmt.Errorf("ssh_helper: error parsing config: %s", err)
	}

	list, ok := root.Node.(*ast.ObjectList)
	if !ok {
		return nil, fmt.Errorf("ssh_helper: error parsing config: file doesn't contain a root object")
	}

	valid := []string{
		"vault_addr",
		"ssh_mount_point",
		"ca_cert",
		"ca_path",
		"allowed_cidr_list",
		"allowed_roles",
		"tls_skip_verify",
		"tls_server_name",
	}
	if err := checkHCLKeys(list, valid); err != nil {
		return nil, multierror.Prefix(err, "ssh_helper:")
	}

	var c SSHHelperConfig
	c.SSHMountPoint = SSHHelperDefaultMountPoint
	if err := hcl.DecodeObject(&c, list); err != nil {
		return nil, multierror.Prefix(err, "ssh_helper:")
	}

	if c.VaultAddr == "" {
		return nil, fmt.Errorf("ssh_helper: missing config 'vault_addr'")
	}
	return &c, nil
}

// SSHHelper creates an SSHHelper object which can talk to Vault server with SSH backend
// mounted at default path ("ssh").
func (c *Client) SSHHelper() *SSHHelper {
	return c.SSHHelperWithMountPoint(SSHHelperDefaultMountPoint)
}

// SSHHelperWithMountPoint creates an SSHHelper object which can talk to Vault server with SSH backend
// mounted at a specific mount point.
func (c *Client) SSHHelperWithMountPoint(mountPoint string) *SSHHelper {
	return &SSHHelper{
		c:          c,
		MountPoint: mountPoint,
	}
}

// Verify verifies if the key provided by user is present in Vault server. The response
// will contain the IP address and username associated with the OTP. In case the
// OTP matches the echo request message, instead of searching an entry for the OTP,
// an echo response message is returned. This feature is used by ssh-helper to verify if
// its configured correctly.
func (c *SSHHelper) Verify(otp string) (*SSHVerifyResponse, error) {
	data := map[string]interface{}{
		"otp": otp,
	}
	verifyPath := fmt.Sprintf("/v1/%s/verify", c.MountPoint)
	r := c.c.NewRequest("PUT", verifyPath)
	if err := r.SetJSONBody(data); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	secret, err := ParseSecret(resp.Body)
	if err != nil {
		return nil, err
	}

	if secret.Data == nil {
		return nil, nil
	}

	var verifyResp SSHVerifyResponse
	err = mapstructure.Decode(secret.Data, &verifyResp)
	if err != nil {
		return nil, err
	}
	return &verifyResp, nil
}

func checkHCLKeys(node ast.Node, valid []string) error {
	var list *ast.ObjectList
	switch n := node.(type) {
	case *ast.ObjectList:
		list = n
	case *ast.ObjectType:
		list = n.List
	default:
		return fmt.Errorf("cannot check HCL keys of type %T", n)
	}

	validMap := make(map[string]struct{}, len(valid))
	for _, v := range valid {
		validMap[v] = struct{}{}
	}

	var result error
	for _, item := range list.Items {
		key := item.Keys[0].Token.Value().(string)
		if _, ok := validMap[key]; !ok {
			result = multierror.Append(result, fmt.Errorf(
				"invalid key '%s' on line %d", key, item.Assign.Line))
		}
	}

	return result
}
