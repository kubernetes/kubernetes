# JOSE CLI

The `jose-util` command line utility allows for encryption, decryption, signing
and verification of JOSE messages. Its main purpose is to facilitate dealing
with JOSE messages when testing or debugging.

## Usage

The utility includes the subcommands `encrypt`, `decrypt`, `sign`, `verify` and
`expand`. Examples for each command can be found below.

Algorithms are selected via the `--alg` and `--enc` flags, which influence the
`alg` and `enc` headers in respectively. For JWE, `--alg` specifies the key
managment algorithm (e.g. `RSA-OAEP`) and `--enc` specifies the content
encryption algorithm (e.g. `A128GCM`). For JWS, `--alg` specifies the
signature algorithm (e.g. `PS256`).

Input and output files can be specified via the `--in` and `--out` flags.
Either flag can be omitted, in which case `jose-util` uses stdin/stdout for
input/output respectively. By default each command will output a compact
message, but it's possible to get the full serialization by supplying the
`--full` flag.

Keys are specified via the `--key` flag. Supported key types are naked RSA/EC
keys and X.509 certificates with embedded RSA/EC keys. Keys must be in PEM
or DER formats.

## Examples

### Encrypt

Takes a plaintext as input, encrypts, and prints the encrypted message.

    jose-util encrypt -k public-key.pem --alg RSA-OAEP --enc A128GCM

### Decrypt

Takes an encrypted message (JWE) as input, decrypts, and prints the plaintext.

    jose-util decrypt -k private-key.pem

### Sign

Takes a payload as input, signs it, and prints the signed message with the embedded payload.

    jose-util sign -k private-key.pem --alg PS256

### Verify

Reads a signed message (JWS), verifies it, and extracts the payload.

    jose-util verify -k public-key.pem

### Expand

Expands a compact message to the full serialization format.

    jose-util expand --format JWE   # Expands a compact JWE to full format
    jose-util expand --format JWS   # Expands a compact JWS to full format
