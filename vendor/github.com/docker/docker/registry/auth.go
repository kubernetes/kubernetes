package registry

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/cliconfig"
)

// Login tries to register/login to the registry server.
func Login(authConfig *cliconfig.AuthConfig, registryEndpoint *Endpoint) (string, error) {
	// Separates the v2 registry login logic from the v1 logic.
	if registryEndpoint.Version == APIVersion2 {
		return loginV2(authConfig, registryEndpoint, "" /* scope */)
	}
	return loginV1(authConfig, registryEndpoint)
}

// loginV1 tries to register/login to the v1 registry server.
func loginV1(authConfig *cliconfig.AuthConfig, registryEndpoint *Endpoint) (string, error) {
	var (
		status        string
		reqBody       []byte
		err           error
		reqStatusCode = 0
		serverAddress = authConfig.ServerAddress
	)

	logrus.Debugf("attempting v1 login to registry endpoint %s", registryEndpoint)

	if serverAddress == "" {
		return "", fmt.Errorf("Server Error: Server Address not set.")
	}

	loginAgainstOfficialIndex := serverAddress == INDEXSERVER

	// to avoid sending the server address to the server it should be removed before being marshalled
	authCopy := *authConfig
	authCopy.ServerAddress = ""

	jsonBody, err := json.Marshal(authCopy)
	if err != nil {
		return "", fmt.Errorf("Config Error: %s", err)
	}

	// using `bytes.NewReader(jsonBody)` here causes the server to respond with a 411 status.
	b := strings.NewReader(string(jsonBody))
	req1, err := registryEndpoint.client.Post(serverAddress+"users/", "application/json; charset=utf-8", b)
	if err != nil {
		return "", fmt.Errorf("Server Error: %s", err)
	}
	reqStatusCode = req1.StatusCode
	defer req1.Body.Close()
	reqBody, err = ioutil.ReadAll(req1.Body)
	if err != nil {
		return "", fmt.Errorf("Server Error: [%#v] %s", reqStatusCode, err)
	}

	if reqStatusCode == 201 {
		if loginAgainstOfficialIndex {
			status = "Account created. Please use the confirmation link we sent" +
				" to your e-mail to activate it."
		} else {
			// *TODO: Use registry configuration to determine what this says, if anything?
			status = "Account created. Please see the documentation of the registry " + serverAddress + " for instructions how to activate it."
		}
	} else if reqStatusCode == 400 {
		if string(reqBody) == "\"Username or email already exists\"" {
			req, err := http.NewRequest("GET", serverAddress+"users/", nil)
			req.SetBasicAuth(authConfig.Username, authConfig.Password)
			resp, err := registryEndpoint.client.Do(req)
			if err != nil {
				return "", err
			}
			defer resp.Body.Close()
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				return "", err
			}
			if resp.StatusCode == 200 {
				return "Login Succeeded", nil
			} else if resp.StatusCode == 401 {
				return "", fmt.Errorf("Wrong login/password, please try again")
			} else if resp.StatusCode == 403 {
				if loginAgainstOfficialIndex {
					return "", fmt.Errorf("Login: Account is not Active. Please check your e-mail for a confirmation link.")
				}
				// *TODO: Use registry configuration to determine what this says, if anything?
				return "", fmt.Errorf("Login: Account is not Active. Please see the documentation of the registry %s for instructions how to activate it.", serverAddress)
			} else if resp.StatusCode == 500 { // Issue #14326
				logrus.Errorf("%s returned status code %d. Response Body :\n%s", req.URL.String(), resp.StatusCode, body)
				return "", fmt.Errorf("Internal Server Error")
			}
			return "", fmt.Errorf("Login: %s (Code: %d; Headers: %s)", body, resp.StatusCode, resp.Header)
		}
		return "", fmt.Errorf("Registration: %s", reqBody)

	} else if reqStatusCode == 401 {
		// This case would happen with private registries where /v1/users is
		// protected, so people can use `docker login` as an auth check.
		req, err := http.NewRequest("GET", serverAddress+"users/", nil)
		req.SetBasicAuth(authConfig.Username, authConfig.Password)
		resp, err := registryEndpoint.client.Do(req)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}
		if resp.StatusCode == 200 {
			return "Login Succeeded", nil
		} else if resp.StatusCode == 401 {
			return "", fmt.Errorf("Wrong login/password, please try again")
		} else {
			return "", fmt.Errorf("Login: %s (Code: %d; Headers: %s)", body,
				resp.StatusCode, resp.Header)
		}
	} else {
		return "", fmt.Errorf("Unexpected status code [%d] : %s", reqStatusCode, reqBody)
	}
	return status, nil
}

// loginV2 tries to login to the v2 registry server. The given registry endpoint has been
// pinged or setup with a list of authorization challenges. Each of these challenges are
// tried until one of them succeeds. Currently supported challenge schemes are:
// 		HTTP Basic Authorization
// 		Token Authorization with a separate token issuing server
// NOTE: the v2 logic does not attempt to create a user account if one doesn't exist. For
// now, users should create their account through other means like directly from a web page
// served by the v2 registry service provider. Whether this will be supported in the future
// is to be determined.
func loginV2(authConfig *cliconfig.AuthConfig, registryEndpoint *Endpoint, scope string) (string, error) {
	logrus.Debugf("attempting v2 login to registry endpoint %s", registryEndpoint)
	var (
		err       error
		allErrors []error
	)

	for _, challenge := range registryEndpoint.AuthChallenges {
		params := make(map[string]string, len(challenge.Parameters)+1)
		for k, v := range challenge.Parameters {
			params[k] = v
		}
		params["scope"] = scope
		logrus.Debugf("trying %q auth challenge with params %v", challenge.Scheme, params)

		switch strings.ToLower(challenge.Scheme) {
		case "basic":
			err = tryV2BasicAuthLogin(authConfig, params, registryEndpoint)
		case "bearer":
			err = tryV2TokenAuthLogin(authConfig, params, registryEndpoint)
		default:
			// Unsupported challenge types are explicitly skipped.
			err = fmt.Errorf("unsupported auth scheme: %q", challenge.Scheme)
		}

		if err == nil {
			return "Login Succeeded", nil
		}

		logrus.Debugf("error trying auth challenge %q: %s", challenge.Scheme, err)

		allErrors = append(allErrors, err)
	}

	return "", fmt.Errorf("no successful auth challenge for %s - errors: %s", registryEndpoint, allErrors)
}

func tryV2BasicAuthLogin(authConfig *cliconfig.AuthConfig, params map[string]string, registryEndpoint *Endpoint) error {
	req, err := http.NewRequest("GET", registryEndpoint.Path(""), nil)
	if err != nil {
		return err
	}

	req.SetBasicAuth(authConfig.Username, authConfig.Password)

	resp, err := registryEndpoint.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("basic auth attempt to %s realm %q failed with status: %d %s", registryEndpoint, params["realm"], resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	return nil
}

func tryV2TokenAuthLogin(authConfig *cliconfig.AuthConfig, params map[string]string, registryEndpoint *Endpoint) error {
	token, err := getToken(authConfig.Username, authConfig.Password, params, registryEndpoint)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("GET", registryEndpoint.Path(""), nil)
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	resp, err := registryEndpoint.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("token auth attempt to %s realm %q failed with status: %d %s", registryEndpoint, params["realm"], resp.StatusCode, http.StatusText(resp.StatusCode))
	}

	return nil
}

// this method matches a auth configuration to a server address or a url
func ResolveAuthConfig(config *cliconfig.ConfigFile, index *IndexInfo) cliconfig.AuthConfig {
	configKey := index.GetAuthConfigKey()
	// First try the happy case
	if c, found := config.AuthConfigs[configKey]; found || index.Official {
		return c
	}

	convertToHostname := func(url string) string {
		stripped := url
		if strings.HasPrefix(url, "http://") {
			stripped = strings.Replace(url, "http://", "", 1)
		} else if strings.HasPrefix(url, "https://") {
			stripped = strings.Replace(url, "https://", "", 1)
		}

		nameParts := strings.SplitN(stripped, "/", 2)

		return nameParts[0]
	}

	// Maybe they have a legacy config file, we will iterate the keys converting
	// them to the new format and testing
	for registry, ac := range config.AuthConfigs {
		if configKey == convertToHostname(registry) {
			return ac
		}
	}

	// When all else fails, return an empty auth config
	return cliconfig.AuthConfig{}
}
