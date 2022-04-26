package ssh_config

import (
	"fmt"
	"strconv"
	"strings"
)

// Default returns the default value for the given keyword, for example "22" if
// the keyword is "Port". Default returns the empty string if the keyword has no
// default, or if the keyword is unknown. Keyword matching is case-insensitive.
//
// Default values are provided by OpenSSH_7.4p1 on a Mac.
func Default(keyword string) string {
	return defaults[strings.ToLower(keyword)]
}

// Arguments where the value must be "yes" or "no" and *only* yes or no.
var yesnos = map[string]bool{
	strings.ToLower("BatchMode"):                        true,
	strings.ToLower("CanonicalizeFallbackLocal"):        true,
	strings.ToLower("ChallengeResponseAuthentication"):  true,
	strings.ToLower("CheckHostIP"):                      true,
	strings.ToLower("ClearAllForwardings"):              true,
	strings.ToLower("Compression"):                      true,
	strings.ToLower("EnableSSHKeysign"):                 true,
	strings.ToLower("ExitOnForwardFailure"):             true,
	strings.ToLower("ForwardAgent"):                     true,
	strings.ToLower("ForwardX11"):                       true,
	strings.ToLower("ForwardX11Trusted"):                true,
	strings.ToLower("GatewayPorts"):                     true,
	strings.ToLower("GSSAPIAuthentication"):             true,
	strings.ToLower("GSSAPIDelegateCredentials"):        true,
	strings.ToLower("HostbasedAuthentication"):          true,
	strings.ToLower("IdentitiesOnly"):                   true,
	strings.ToLower("KbdInteractiveAuthentication"):     true,
	strings.ToLower("NoHostAuthenticationForLocalhost"): true,
	strings.ToLower("PasswordAuthentication"):           true,
	strings.ToLower("PermitLocalCommand"):               true,
	strings.ToLower("PubkeyAuthentication"):             true,
	strings.ToLower("RhostsRSAAuthentication"):          true,
	strings.ToLower("RSAAuthentication"):                true,
	strings.ToLower("StreamLocalBindUnlink"):            true,
	strings.ToLower("TCPKeepAlive"):                     true,
	strings.ToLower("UseKeychain"):                      true,
	strings.ToLower("UsePrivilegedPort"):                true,
	strings.ToLower("VisualHostKey"):                    true,
}

var uints = map[string]bool{
	strings.ToLower("CanonicalizeMaxDots"):     true,
	strings.ToLower("CompressionLevel"):        true, // 1 to 9
	strings.ToLower("ConnectionAttempts"):      true,
	strings.ToLower("ConnectTimeout"):          true,
	strings.ToLower("NumberOfPasswordPrompts"): true,
	strings.ToLower("Port"):                    true,
	strings.ToLower("ServerAliveCountMax"):     true,
	strings.ToLower("ServerAliveInterval"):     true,
}

func mustBeYesOrNo(lkey string) bool {
	return yesnos[lkey]
}

func mustBeUint(lkey string) bool {
	return uints[lkey]
}

func validate(key, val string) error {
	lkey := strings.ToLower(key)
	if mustBeYesOrNo(lkey) && (val != "yes" && val != "no") {
		return fmt.Errorf("ssh_config: value for key %q must be 'yes' or 'no', got %q", key, val)
	}
	if mustBeUint(lkey) {
		_, err := strconv.ParseUint(val, 10, 64)
		if err != nil {
			return fmt.Errorf("ssh_config: %v", err)
		}
	}
	return nil
}

var defaults = map[string]string{
	strings.ToLower("AddKeysToAgent"):                  "no",
	strings.ToLower("AddressFamily"):                   "any",
	strings.ToLower("BatchMode"):                       "no",
	strings.ToLower("CanonicalizeFallbackLocal"):       "yes",
	strings.ToLower("CanonicalizeHostname"):            "no",
	strings.ToLower("CanonicalizeMaxDots"):             "1",
	strings.ToLower("ChallengeResponseAuthentication"): "yes",
	strings.ToLower("CheckHostIP"):                     "yes",
	// TODO is this still the correct cipher
	strings.ToLower("Cipher"):                    "3des",
	strings.ToLower("Ciphers"):                   "chacha20-poly1305@openssh.com,aes128-ctr,aes192-ctr,aes256-ctr,aes128-gcm@openssh.com,aes256-gcm@openssh.com,aes128-cbc,aes192-cbc,aes256-cbc",
	strings.ToLower("ClearAllForwardings"):       "no",
	strings.ToLower("Compression"):               "no",
	strings.ToLower("CompressionLevel"):          "6",
	strings.ToLower("ConnectionAttempts"):        "1",
	strings.ToLower("ControlMaster"):             "no",
	strings.ToLower("EnableSSHKeysign"):          "no",
	strings.ToLower("EscapeChar"):                "~",
	strings.ToLower("ExitOnForwardFailure"):      "no",
	strings.ToLower("FingerprintHash"):           "sha256",
	strings.ToLower("ForwardAgent"):              "no",
	strings.ToLower("ForwardX11"):                "no",
	strings.ToLower("ForwardX11Timeout"):         "20m",
	strings.ToLower("ForwardX11Trusted"):         "no",
	strings.ToLower("GatewayPorts"):              "no",
	strings.ToLower("GlobalKnownHostsFile"):      "/etc/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts2",
	strings.ToLower("GSSAPIAuthentication"):      "no",
	strings.ToLower("GSSAPIDelegateCredentials"): "no",
	strings.ToLower("HashKnownHosts"):            "no",
	strings.ToLower("HostbasedAuthentication"):   "no",

	strings.ToLower("HostbasedKeyTypes"): "ecdsa-sha2-nistp256-cert-v01@openssh.com,ecdsa-sha2-nistp384-cert-v01@openssh.com,ecdsa-sha2-nistp521-cert-v01@openssh.com,ssh-ed25519-cert-v01@openssh.com,ssh-rsa-cert-v01@openssh.com,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521,ssh-ed25519,ssh-rsa",
	strings.ToLower("HostKeyAlgorithms"): "ecdsa-sha2-nistp256-cert-v01@openssh.com,ecdsa-sha2-nistp384-cert-v01@openssh.com,ecdsa-sha2-nistp521-cert-v01@openssh.com,ssh-ed25519-cert-v01@openssh.com,ssh-rsa-cert-v01@openssh.com,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521,ssh-ed25519,ssh-rsa",
	// HostName has a dynamic default (the value passed at the command line).

	strings.ToLower("IdentitiesOnly"): "no",
	strings.ToLower("IdentityFile"):   "~/.ssh/identity",

	// IPQoS has a dynamic default based on interactive or non-interactive
	// sessions.

	strings.ToLower("KbdInteractiveAuthentication"): "yes",

	strings.ToLower("KexAlgorithms"): "curve25519-sha256,curve25519-sha256@libssh.org,ecdh-sha2-nistp256,ecdh-sha2-nistp384,ecdh-sha2-nistp521,diffie-hellman-group-exchange-sha256,diffie-hellman-group-exchange-sha1,diffie-hellman-group14-sha1",
	strings.ToLower("LogLevel"):      "INFO",
	strings.ToLower("MACs"):          "umac-64-etm@openssh.com,umac-128-etm@openssh.com,hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha1-etm@openssh.com,umac-64@openssh.com,umac-128@openssh.com,hmac-sha2-256,hmac-sha2-512,hmac-sha1",

	strings.ToLower("NoHostAuthenticationForLocalhost"): "no",
	strings.ToLower("NumberOfPasswordPrompts"):          "3",
	strings.ToLower("PasswordAuthentication"):           "yes",
	strings.ToLower("PermitLocalCommand"):               "no",
	strings.ToLower("Port"):                             "22",

	strings.ToLower("PreferredAuthentications"): "gssapi-with-mic,hostbased,publickey,keyboard-interactive,password",
	strings.ToLower("Protocol"):                 "2",
	strings.ToLower("ProxyUseFdpass"):           "no",
	strings.ToLower("PubkeyAcceptedKeyTypes"):   "ecdsa-sha2-nistp256-cert-v01@openssh.com,ecdsa-sha2-nistp384-cert-v01@openssh.com,ecdsa-sha2-nistp521-cert-v01@openssh.com,ssh-ed25519-cert-v01@openssh.com,ssh-rsa-cert-v01@openssh.com,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521,ssh-ed25519,ssh-rsa",
	strings.ToLower("PubkeyAuthentication"):     "yes",
	strings.ToLower("RekeyLimit"):               "default none",
	strings.ToLower("RhostsRSAAuthentication"):  "no",
	strings.ToLower("RSAAuthentication"):        "yes",

	strings.ToLower("ServerAliveCountMax"):   "3",
	strings.ToLower("ServerAliveInterval"):   "0",
	strings.ToLower("StreamLocalBindMask"):   "0177",
	strings.ToLower("StreamLocalBindUnlink"): "no",
	strings.ToLower("StrictHostKeyChecking"): "ask",
	strings.ToLower("TCPKeepAlive"):          "yes",
	strings.ToLower("Tunnel"):                "no",
	strings.ToLower("TunnelDevice"):          "any:any",
	strings.ToLower("UpdateHostKeys"):        "no",
	strings.ToLower("UseKeychain"):           "no",
	strings.ToLower("UsePrivilegedPort"):     "no",

	strings.ToLower("UserKnownHostsFile"): "~/.ssh/known_hosts ~/.ssh/known_hosts2",
	strings.ToLower("VerifyHostKeyDNS"):   "no",
	strings.ToLower("VisualHostKey"):      "no",
	strings.ToLower("XAuthLocation"):      "/usr/X11R6/bin/xauth",
}
