// Package transport implements functions for facilitating proper TLS-secured
// communications for clients and servers.
//
// Clients should build an identity (of the core.identity) type, such as
//
//     var id = &core.Identity{
//              Request: &csr.CertificateRequest{
//                      CN: "localhost test certificate",
//              },
//              Profiles: map[string]map[string]string{
//                      "paths": map[string]string{
//                              "private_key": "client.key",
//                              "certificate": "client.pem",
//                      },
//                      "cfssl": {
//                              "label":     "",
//                              "profile":   "client-ca",
//                              "remote":    "ca.example.net",
//                              "auth-type": "standard",
//                              "auth-key":  "000102030405060708090a0b0c0d0e0f",
//                      },
//              },
//     }
//
//
//
// The New function will return a transport built using the
// NewKeyProvider and NewCA functions. These functions may be changed
// by other packages to provide common key provider and CA
// configurations. Clients can then use RefreshKeys (or launch
// AutoUpdate in a goroutine) to ensure the certificate and key are
// loaded and correct. The Listen and Dial functions then provide the
// necessary connection support.
//
// The AutoUpdate function will handle automatic certificate
// issuance. Servers and clients are not required to take any special
// action when the certificate is updated: the key and certificate are
// only used when establishing a connection, and therefore existing
// connections are not affected---there is no need to reset or restart
// any existing connections. Clients should run AutoUpdate if they
// plan on making multiple connections or will be reconnecting; for a
// one-off connection, it isn't necessary.
package transport
