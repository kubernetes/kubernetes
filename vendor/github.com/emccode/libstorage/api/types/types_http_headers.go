package types

// All header names below follow the Golang canonical format for header keys.
// Please do not alter their casing to your liking or you will break stuff.
const (
	// InstanceIDHeader is the HTTP header that contains an InstanceID.
	InstanceIDHeader = "Libstorage-Instanceid"

	// LocalDevicesHeader is the HTTP header that contains a local device pair.
	LocalDevicesHeader = "Libstorage-Localdevices"

	// TransactionHeader is the HTTP header that contains the transaction
	// sent from the client.
	TransactionHeader = "Libstorage-Tx"

	// ServerNameHeader is the HTTP header that contains the randomly generated
	// name the server creates for unique identification when the server starts
	// for the first time. This header is provided with every response sent
	// from the server.
	ServerNameHeader = "Libstorage-Servername"
)
