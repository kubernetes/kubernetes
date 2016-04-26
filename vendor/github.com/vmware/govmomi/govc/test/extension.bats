#!/usr/bin/env bats

load test_helper

@test "extension" {
  vcsim_env

  govc extension.info | grep Name: | grep govc-test | awk '{print $2}' | $xargs -r govc extension.unregister

  run govc extension.info enoent
  assert_failure

  id=$(new_id)

  result=$(govc extension.info | grep $id | wc -l)
  [ $result -eq 0 ]

  # register extension
  run govc extension.register $id <<EOS
  {
    "Description": {
      "Label": "govc",
      "Summary": "Go interface to vCenter"
    },
    "Key": "${id}",
    "Company": "VMware, Inc.",
    "Version": "0.2.0"
  }
EOS
  assert_success

  # check info output is legit
  run govc extension.info $id
  assert_line "Name: $id"

  json=$(govc extension.info -json $id)
  label=$(jq -r .Extensions[].Description.Label <<<"$json")
  assert_equal "govc" "$label"

  # change label and update extension
  json=$(jq -r '.Extensions[] | .Description.Label = "novc"' <<<"$json")
  run govc extension.register -update $id <<<"$json"
  assert_success

  # check label changed in info output
  json=$(govc extension.info -json $id)
  label=$(jq -r .Extensions[].Description.Label <<<"$json")
  assert_equal "novc" "$label"

  # set extension certificate to generated certificate
  run govc extension.setcert -cert-pem '+' $id
  assert_success

  # test client certificate authentication
  (
    # remove password from env, set user to extension id and turn of session cache
    govc_url_to_vars
    unset GOVC_PASSWORD
    GOVC_USERNAME=$id
    export GOVC_PERSIST_SESSION=false
    # vagrant port forwards to VC's port 80
    export GOVC_TUNNEL_PROXY_PORT=16080
    run govc about -cert "${id}.crt" -key "${id}.key"
    assert_success
  )

  # remove generated cert and key
  rm ${id}.{crt,key}

  run govc extension.unregister $id
  assert_success

  result=$(govc extension.info | grep $id | wc -l)
  [ $result -eq 0 ]
}
