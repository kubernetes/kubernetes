# Replace with your public address and port for keystone
export OS_AUTH_URL="{{ protocol }}://{{ address }}:{{ port }}/v{{ version }}"
#export OS_PROJECT_NAME=k8s
#export OS_DOMAIN_NAME=k8s
#export OS_USERNAME=myuser
#export OS_PASSWORD=secure_pw
get_keystone_token() {
  data='{ "auth": {
    "identity": {
      "methods": ["password"],
      "password": {
        "user": {
          "name": "'"${OS_USERNAME}"'",
          "domain": { "name": "'"${OS_DOMAIN_NAME}"'" },
          "password": "'"${OS_PASSWORD}"'"
        }
      }
    }
  }
}'
  token=$(curl -s -i -H "Content-Type: application/json" -d "${data}" "${OS_AUTH_URL}/auth/tokens" |grep 'X-Subject-Token')
  if [ -z "$token" ]; then
    echo "Invalid authentication information"
  else
    echo $(echo ${token} | awk -F ': ' '{print $2}')
  fi
}
echo "Function get_keystone_token created. Type get_keystone_token in order to generate a login token for the Kubernetes dashboard."

