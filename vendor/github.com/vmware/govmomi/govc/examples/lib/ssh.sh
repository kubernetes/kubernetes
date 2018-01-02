function public-key {
  local dir=${HOME}/.ssh

  for f in $HOME/.ssh/{id_{rsa,dsa},*}.pub; do
    if [ -r $f ]; then
      echo $f
      return
    fi
  done

  echo "Can't find public key file..."
  exit 1
}

PUBLIC_KEY_FILE=${PUBLIC_KEY_FILE-$(public-key)}
SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=quiet"

function upload-public-key {
  local vm_name=$1
  local dir=$2

  if [ -z "$dir" ]
  then
    uid=$(echo $GOVC_GUEST_LOGIN | awk -F: '{print $1}')
    dir=$(govc guest.getenv -vm ${vm_name} HOME | awk -F= '{print $2}')

    if [ -z "$dir" ]
    then
      echo "Can't find ${uid}'s HOME dir..."
      exit 1
    fi
  fi

  govc guest.mkdir \
       -vm ${vm_name} \
       -p \
       ${dir}/.ssh

  govc guest.upload \
       -vm ${vm_name} \
       -f \
       ${PUBLIC_KEY_FILE} \
       ${dir}/.ssh/authorized_keys
}
