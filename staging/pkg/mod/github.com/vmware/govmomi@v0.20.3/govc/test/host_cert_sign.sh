#!/bin/bash -e

# Usage: govc host.cert.csr | ./host_cert_sign.sh | govc host.cert.import

pushd "$(dirname "$0")" >/dev/null

days=$((365 * 5))

if [ ! -e govc_ca.key ] ; then
  echo "Generating CA private key..." 1>&2
  openssl genrsa -out govc_ca.key 2048

  echo "Generating CA self signed certificate..." 1>&2
  openssl req -x509 -new -nodes -key govc_ca.key -out govc_ca.pem -subj /C=US/ST=CA/L=SF/O=VMware/OU=Eng/CN=govc-ca -days $days
fi

echo "Signing CSR with the CA certificate..." 1>&2

# The hostd generated CSR includes:
#   Requested Extensions:
#       X509v3 Subject Alternative Name:
#       IP Address:$ip
# But seems it doesn't get copied by default, so we end up with:
#   x509: cannot validate certificate for $ip because it doesn't contain any IP SANs (x509.HostnameError)
# Using -extfile to add it to the signed cert.

ip=$(govc env -x GOVC_URL_HOST)
openssl x509 -req -CA govc_ca.pem -CAkey govc_ca.key -CAcreateserial -days $days -extfile <(echo "subjectAltName=IP:$ip")

popd >/dev/null
