#!/bin/bash -e
#
# Capture ESXi traffic and decrypt SOAP traffic on port 443 via wireshark

# Device to capture
dev="${1-vmk0}"

# Device to get the ip for wireshark ssl_keys config
if [ "$dev" = "lo0" ] ; then
  ip_dev="vmk0"
else
  ip_dev="$dev"
fi

ip=$(govc host.info -k -json | \
        jq -r ".HostSystems[].Config.Network.Vnic[] | select(.Device == \"${ip_dev}\") | .Spec.Ip.IpAddress")

scp=(scp)
ssh=(ssh)

# Check if vagrant ssh-config applies to $ip
if [ -d ".vagrant" ] ; then
  vssh_opts=($(vagrant ssh-config | awk NF | awk -v ORS=' ' '{print "-o " $1 "=" $2}'))
  if grep "HostName=${ip}" >/dev/null <<<"${vssh_opts[*]}" ; then
    ssh_opts=("${vssh_opts[@]}")
  fi
fi

# Otherwise, use default ssh opts + sshpass if available
if [ ${#ssh_opts[@]} -eq 0 ] ; then
  user="$(govc env GOVC_USERNAME)"
  ssh_opts=(-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=FATAL -o User=$user)

  if [ -x "$(which sshpass)" ] ; then
    password="$(govc env GOVC_PASSWORD)"
    scp=(sshpass -p $password scp)
    ssh=(sshpass -p $password ssh)
  fi
fi

if [ "$dev" != "lo0" ] ; then
  # If you change this filter, be sure to exclude the ssh port (not tcp port 22)
  filter="host $ip and \(port 80 or port 443\)"

  dst="$HOME/.wireshark/rui-${ip}.key"
  if [ ! -f "$dst" ] ; then
    # Copy key from ESX
    "${scp[@]}" "${ssh_opts[@]}" "${ip}:/etc/vmware/ssl/rui.key" "$dst"
  fi

  if ! grep "$ip" ~/.wireshark/ssl_keys 2>/dev/null ; then
    # Add key to wireshark ssl_keys config
    echo "adding rui.key for $ip"

    cat <<EOF >> ~/.wireshark/ssl_keys
"$ip","443","http","$dst",""
EOF
  fi
fi

echo "Capturing $dev on $ip..."

"${ssh[@]}" "${ssh_opts[@]}" "$ip" tcpdump-uw -i "$dev" -s0 -v -w - "$filter" | wireshark -k -i -
