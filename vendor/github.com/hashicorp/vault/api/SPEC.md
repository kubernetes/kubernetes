FORMAT: 1A

# vault

The Vault API gives you full access to the Vault project.

If you're browsing this API specifiction in GitHub or in raw
format, please excuse some of the odd formatting. This document
is in api-blueprint format that is read by viewers such as
Apiary.

## Sealed vs. Unsealed

Whenever an individual Vault server is started, it is started
in the _sealed_ state. In this state, it knows where its data
is located, but the data is encrypted and Vault doesn't have the
encryption keys to access it. Before Vault can operate, it must
be _unsealed_.

**Note:** Sealing/unsealing has no relationship to _authentication_
which is separate and still required once the Vault is unsealed.

Instead of being sealed with a single key, we utilize
[Shamir's Secret Sharing](http://en.wikipedia.org/wiki/Shamir%27s_Secret_Sharing)
to shard a key into _n_ parts such that _t_ parts are required
to reconstruct the original key, where `t <= n`. This means that
Vault itself doesn't know the original key, and no single person
has the original key (unless `n = 1`, or `t` parts are given to
a single person).

Unsealing is done via an unauthenticated
[unseal API](#reference/seal/unseal/unseal). This API takes a single
master shard and progresses the unsealing process. Once all shards
are given, the Vault is either unsealed or resets the unsealing
process if the key was invalid.

The entire seal/unseal state is server-wide. This allows multiple
distinct operators to use the unseal API (or more likely the
`vault unseal` command) from separate computers/networks and never
have to transmit their key in order to unseal the vault in a
distributed fashion.

## Transport

The API is expected to be accessed over a TLS connection at
all times, with a valid certificate that is verified by a well
behaved client.

## Authentication

Once the Vault is unsealed, every other operation requires
authentication. There are multiple methods for authentication
that can be enabled (see
[authentication](#reference/authentication)).

Authentication is done with the login endpoint. The login endpoint
returns an access token that is set as the `X-Vault-Token` header.

## Help

To retrieve the help for any API within Vault, including mounted
backends, credential providers, etc. then append `?help=1` to any
URL. If you have valid permission to access the path, then the help text
will be returned with the following structure:

        {
            "help": "help text"
        }

## Error Response

A common JSON structure is always returned to return errors:

        {
            "errors": [
                "message",
                "another message"
            ]
        }

This structure will be sent down for any non-20x HTTP status.

## HTTP Status Codes

The following HTTP status codes are used throughout the API.

- `200` - Success with data.
- `204` - Success, no data returned.
- `400` - Invalid request, missing or invalid data.
- `403` - Forbidden, your authentication details are either
   incorrect or you don't have access to this feature.
- `404` - Invalid path. This can both mean that the path truly
   doesn't exist or that you don't have permission to view a
   specific path. We use 404 in some cases to avoid state leakage.
- `429` - Rate limit exceeded. Try again after waiting some period
   of time.
- `500` - Internal server error. An internal error has occurred,
   try again later. If the error persists, report a bug.
- `503` - Vault is down for maintenance or is currently sealed.
   Try again later.

# Group Initialization

## Initialization [/sys/init]
### Initialization Status [GET]
Returns the status of whether the vault is initialized or not. The
vault doesn't have to be unsealed for this operation.

+ Response 200 (application/json)

        {
            "initialized": true
        }

### Initialize [POST]
Initialize the vault. This is an unauthenticated request to initially
setup a new vault. Although this is unauthenticated, it is still safe:
data cannot be in vault prior to initialization, and any future
authentication will fail if you didn't initialize it yourself.
Additionally, once initialized, a vault cannot be reinitialized.

This API is the only time Vault will ever be aware of your keys, and
the only time the keys will ever be returned in one unit. Care should
be taken to ensure that the output of this request is never logged,
and that the keys are properly distributed.

The response also contains the initial root token that can be used
as authentication in order to initially configure Vault once it is
unsealed. Just as with the unseal keys, this is the only time Vault is
ever aware of this token.

+ Request (application/json)

        {
            "secret_shares": 5,
            "secret_threshold": 3,
        }

+ Response 200 (application/json)

        {
            "keys": ["one", "two", "three"],
            "root_token": "foo"
        }

# Group Seal/Unseal

## Seal Status [/sys/seal-status]
### Seal Status [GET]
Returns the status of whether the vault is currently
sealed or not, as well as the progress of unsealing.

The response has the following attributes:

- sealed (boolean) - If true, the vault is sealed. Otherwise,
    it is unsealed.
- t (int) - The "t" value for the master key, or the number
    of shards needed total to unseal the vault.
- n (int) - The "n" value for the master key, or the total
    number of shards of the key distributed.
- progress (int) - The number of master key shards that have
    been entered so far towards unsealing the vault.

+ Response 200 (application/json)

        {
            "sealed": true,
            "t": 3,
            "n": 5,
            "progress": 1
        }

## Seal [/sys/seal]
### Seal [PUT]
Seal the vault.

Sealing the vault locks Vault from any future operations on any
secrets or system configuration until the vault is once again
unsealed. Internally, sealing throws away the keys to access the
encrypted vault data, so Vault is unable to access the data without
unsealing to get the encryption keys.

+ Response 204

## Unseal [/sys/unseal]
### Unseal [PUT]
Unseal the vault.

Unseal the vault by entering a portion of the master key. The
response object will tell you if the unseal is complete or
only partial.

If the vault is already unsealed, this does nothing. It is
not an error, the return value just says the vault is unsealed.
Due to the architecture of Vault, we cannot validate whether
any portion of the unseal key given is valid until all keys
are inputted, therefore unsealing an already unsealed vault
is still a success even if the input key is invalid.

+ Request (application/json)

        {
            "key": "value"
        }

+ Response 200 (application/json)

        {
            "sealed": true,
            "t": 3,
            "n": 5,
            "progress": 1
        }

# Group Authentication

## List Auth Methods [/sys/auth]
### List all auth methods [GET]
Lists all available authentication methods.

This returns the name of the authentication method as well as
a human-friendly long-form help text for the method that can be
shown to the user as documentation.

+ Response 200 (application/json)

        {
            "token": {
                "type": "token",
                "description": "Token authentication"
            },
            "oauth": {
                "type": "oauth",
                "description": "OAuth authentication"
            }
        }

## Single Auth Method [/sys/auth/{id}]

+ Parameters
    + id (required, string) ... The ID of the auth method.

### Enable an auth method [PUT]
Enables an authentication method.

The body of the request depends on the authentication method
being used. Please reference the documentation for the specific
authentication method you're enabling in order to determine what
parameters you must give it.

If an authentication method is already enabled, then this can be
used to change the configuration, including even the type of
the configuration.

+ Request (application/json)

        {
            "type": "type",
            "key": "value",
            "key2": "value2"
        }

+ Response 204

### Disable an auth method [DELETE]
Disables an authentication method. Previously authenticated sessions
are immediately invalidated.

+ Response 204

# Group Policies

Policies are named permission sets that identities returned by
credential stores are bound to. This separates _authentication_
from _authorization_.

## Policies [/sys/policy]
### List all Policies [GET]

List all the policies.

+ Response 200 (application/json)

        {
            "policies": ["root"]
        }

## Single Policy [/sys/policy/{id}]

+ Parameters
    + id (required, string) ... The name of the policy

### Upsert [PUT]

Create or update a policy with the given ID.

+ Request (application/json)

        {
            "rules": "HCL"
        }

+ Response 204

### Delete [DELETE]

Delete a policy with the given ID. Any identities bound to this
policy will immediately become "deny all" despite already being
authenticated.

+ Response 204

# Group Mounts

Logical backends are mounted at _mount points_, similar to
filesystems. This allows you to mount the "aws" logical backend
at the "aws-us-east" path, so all access is at `/aws-us-east/keys/foo`
for example. This enables multiple logical backends to be enabled.

## Mounts [/sys/mounts]
### List all mounts [GET]

Lists all the active mount points.

+ Response 200 (application/json)

        {
            "aws": {
                "type": "aws",
                "description": "AWS"
            },
            "pg": {
                "type": "postgresql",
                "description": "PostgreSQL dynamic users"
            }
        }

## Single Mount [/sys/mounts/{path}]
### New Mount [POST]

Mount a logical backend to a new path.

Configuration for this new backend is done via the normal
read/write mechanism once it is mounted.

+ Request (application/json)

        {
            "type": "aws",
            "description": "EU AWS tokens"
        }

+ Response 204

### Unmount [DELETE]

Unmount a mount point.

+ Response 204

## Remount [/sys/remount]
### Remount [POST]

Move an already-mounted backend to a new path.

+ Request (application/json)

        {
            "from": "aws",
            "to": "aws-east"
        }

+ Response 204

# Group Audit Backends

Audit backends are responsible for shuttling the audit logs that
Vault generates to a durable system for future querying. By default,
audit logs are not stored anywhere.

## Audit Backends [/sys/audit]
### List Enabled Audit Backends [GET]

List all the enabled audit backends

+ Response 200 (application/json)

        {
            "file": {
                "type": "file",
                "description": "Send audit logs to a file",
                "options": {}
            }
        }

## Single Audit Backend [/sys/audit/{path}]

+ Parameters
    + path (required, string) ... The path where the audit backend is mounted

### Enable [PUT]

Enable an audit backend.

+ Request (application/json)

        {
            "type": "file",
            "description": "send to a file",
            "options": {
                "path": "/var/log/vault.audit.log"
            }
        }

+ Response 204

### Disable [DELETE]

Disable an audit backend.

+ Request (application/json)

+ Response 204

# Group Secrets

## Generic [/{mount}/{path}]

This group documents the general format of reading and writing
to Vault. The exact structure of the keyspace is defined by the
logical backends in use, so documentation related to
a specific backend should be referenced for details on what keys
and routes are expected.

The path for examples are `/prefix/path`, but in practice
these will be defined by the backends that are mounted. For
example, reading an AWS key might be at the `/aws/root` path.
These paths are defined by the logical backends.

+ Parameters
    + mount (required, string) ... The mount point for the
      logical backend. Example: `aws`.
    + path (optional, string) ... The path within the backend
      to read or write data.

### Read [GET]

Read data from vault.

The data read from the vault can either be a secret or
arbitrary configuration data. The type of data returned
depends on the path, and is defined by the logical backend.

If the return value is a secret, then the return structure
is a mixture of arbitrary key/value along with the following
fields which are guaranteed to exist:

- `lease_id` (string) - A unique ID used for renewal and
  revocation.

- `renewable` (bool) - If true, then this key can be renewed.
  If a key can't be renewed, then a new key must be requested
  after the lease duration period.

- `lease_duration` (int) - The time in seconds that a secret is
  valid for before it must be renewed.

- `lease_duration_max` (int) - The maximum amount of time in
  seconds that a secret is valid for. This will always be
  greater than or equal to `lease_duration`. The difference
  between this and `lease_duration` is an overlap window
  where multiple keys may be valid.

If the return value is not a secret, then the return structure
is an arbitrary JSON object.

+ Response 200 (application/json)

        {
            "lease_id": "UUID",
            "lease_duration": 3600,
            "key": "value"
        }

### Write [PUT]

Write data to vault.

The behavior and arguments to the write are defined by
the logical backend.

+ Request (application/json)

        {
            "key": "value"
        }

+ Response 204

# Group Lease Management

## Renew Key [/sys/renew/{id}]

+ Parameters
    + id (required, string) ... The `lease_id` of the secret
      to renew.

### Renew [PUT]

+ Response 200 (application/json)

        {
            "lease_id": "...",
            "lease_duration": 3600,
            "access_key": "foo",
            "secret_key": "bar"
        }

## Revoke Key [/sys/revoke/{id}]

+ Parameters
    + id (required, string) ... The `lease_id` of the secret
      to revoke.

### Revoke [PUT]

+ Response 204

# Group Backend: AWS

## Root Key [/aws/root]
### Set the Key [PUT]

Set the root key that the logical backend will use to create
new secrets, IAM policies, etc.

+ Request (application/json)

        {
            "access_key": "key",
            "secret_key": "key",
            "region": "us-east-1"
        }

+ Response 204

## Policies [/aws/policies]
### List Policies [GET]

List all the policies that can be used to create keys.

+ Response 200 (application/json)

        [{
            "name": "root",
            "description": "Root access"
        }, {
            "name": "web-deploy",
            "description": "Enough permissions to deploy the web app."
        }]

## Single Policy [/aws/policies/{name}]

+ Parameters
    + name (required, string) ... Name of the policy.

### Read [GET]

Read a policy.

+ Response 200 (application/json)

        {
            "policy": "base64-encoded policy"
        }

### Upsert [PUT]

Create or update a policy.

+ Request (application/json)

        {
            "policy": "base64-encoded policy"
        }

+ Response 204

### Delete [DELETE]

Delete the policy with the given name.

+ Response 204

## Generate Access Keys [/aws/keys/{policy}]
### Create [GET]

This generates a new keypair for the given policy.

+ Parameters
    + policy (required, string) ... The policy under which to create
      the key pair.

+ Response 200 (application/json)

        {
            "lease_id": "...",
            "lease_duration": 3600,
            "access_key": "foo",
            "secret_key": "bar"
        }
