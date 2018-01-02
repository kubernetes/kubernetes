#!/bin/bash -e

for var in $(env | grep GOVC_) ; do
  eval "unset ${var/=*}"
done

common_opts=$(cat <<EOF
  -cert=                    Certificate [GOVC_CERTIFICATE]
  -debug=false              Store debug logs [GOVC_DEBUG]
  -dump=false               Enable output dump
  -json=false               Enable JSON output
  -k=false                  Skip verification of server certificate [GOVC_INSECURE]
  -key=                     Private key [GOVC_PRIVATE_KEY]
  -persist-session=true     Persist session to disk [GOVC_PERSIST_SESSION]
  -tls-ca-certs=            TLS CA certificates file [GOVC_TLS_CA_CERTS]
  -tls-known-hosts=         TLS known hosts file [GOVC_TLS_KNOWN_HOSTS]
  -u=                       ESX or vCenter URL [GOVC_URL]
  -vim-namespace=urn:vim25  Vim namespace [GOVC_VIM_NAMESPACE]
  -vim-version=6.0          Vim version [GOVC_VIM_VERSION]
  -dc=                      Datacenter [GOVC_DATACENTER]
  -host.dns=                Find host by FQDN
  -host.ip=                 Find host by IP address
  -host.ipath=              Find host by inventory path
  -host.uuid=               Find host by UUID
  -vm.dns=                  Find VM by FQDN
  -vm.ip=                   Find VM by IP address
  -vm.ipath=                Find VM by inventory path
  -vm.path=                 Find VM by path to .vmx file
  -vm.uuid=                 Find VM by UUID
EOF
)

cat <<'EOF'
# govc usage

This document is generated from `govc -h` and `govc $cmd -h` commands.

The following common options are filtered out in this document,
but appear via `govc $cmd -h`:

```
EOF

printf "%s\n\`\`\`\n\n" "${common_opts}"

cmds=($(govc -h | grep -v Usage))

opts=($(cut -s -d= -f1 <<<"$common_opts" | xargs -n1 | sed -e 's/^/\\/'))
filter=$(printf "|%s=" "${opts[@]}")

for cmd in "${cmds[@]}" ; do
    printf "## %s\n\n" "$cmd"
    printf "\`\`\`\n"
    govc "$cmd" -h | egrep -v "${filter:1}"
    printf "\`\`\`\n\n"
done
