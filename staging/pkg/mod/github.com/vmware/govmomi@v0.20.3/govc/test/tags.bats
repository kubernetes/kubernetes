#!/usr/bin/env bats

load test_helper

@test "tags.category" {
  vcsim_env
  local output

  run govc tags.category.ls
  assert_success # no categories defined yet

  run govc tags.category.info
  assert_success # no categories defined yet

  run govc tags.category.info enoent
  assert_failure # category does not exist

  category_name=$(new_id)

  run govc tags.category.create -d "Cat in the hat" -m "$category_name"
  assert_success

  category_id="$output"
  run govc tags.category.ls
  assert_success

  run govc tags.category.create -m "$category_name"
  assert_failure # already exists

  run govc tags.category.ls
  assert_line "$category_name"

  id=$(govc tags.category.ls -json | jq -r '.[].id')
  assert_matches "$id" "$category_id"

  run govc tags.category.info "$category_name"
  assert_success

  update_name="${category_name}-update"

  run govc tags.category.update -n "$update_name" -d "Green eggs and ham" "$category_id"
  assert_success

  govc tags.category.info "$update_name" | grep -c eggs

  run govc tags.category.info "$category_name"
  assert_failure # does not exist

  run govc tags.category.rm "$category_name"
  assert_failure # should fail with old name

  run govc tags.category.rm "$update_name"
  assert_success
}

@test "tags" {
  vcsim_env
  local output

  run govc tags.ls
  assert_success # no tags defined yet

  run govc tags.ls -k=false
  assert_failure

  run govc tags.ls -k=false -tls-ca-certs <(govc about.cert -show)
  assert_success

  run govc tags.info
  assert_success # no tags defined yet

  run govc tags.info enoent
  assert_failure # specific tag does not exist

  category_name=$(new_id)
  run govc tags.category.create -m "$category_name"
  assert_success

  category="$output"
  test_name="test_name"

  run govc tags.create -c "$category" $test_name
  assert_success
  tag_id="$output"

  govc tags.ls | grep $test_name

  id=$(govc tags.ls -json | jq -r '.[].id')
  assert_matches "$id" "$tag_id"

  update_name="${test_name}-update"
  run govc tags.update -d "Updated tag" -n "$update_name" "$tag_id"
  assert_success

  govc tags.info
  govc tags.info -C=false
  govc tags.info "$update_name" | grep Updated

  run govc tags.create -c "$category_name" "$(new_id)"
  assert_success

  run govc tags.create -c enoent "$(new_id)"
  assert_failure # category name does not exist

  run govc tags.info enoent
  assert_failure # does not exist
}

@test "tags.association" {
  vcsim_env
  local lines

  category_name=$(new_id)
  run govc tags.category.create -m "$category_name"
  assert_success
  category="$output"

  run govc tags.create -c "$category" "$(new_id)"
  assert_success
  tag=$output

  tag_name=$(govc tags.ls -json | jq -r ".[] | select(.id == \"$tag\") | .name")
  run govc find . -type h
  object=${lines[0]}

  run govc tags.attach "$tag" "$object"
  assert_success

  run govc tags.attached.ls "$tag_name"
  assert_success

  result=$(govc tags.attached.ls -r "$object")
  assert_matches "$result" "$tag_name"

  result=$(govc tags.attached.ls -r -json "$object")
  assert_matches "$tag_name" "$result"

  run govc tags.rm "$tag"
  assert_failure # tags still attached

  run govc tags.detach "$tag" "$object"
  assert_success

  run govc tags.attach "$tag_name" "$object"
  assert_success # attach using name instead of ID

  run govc tags.rm "$tag"
  assert_failure # tags still attached

  run govc tags.detach -c enoent "$tag_name" "$object"
  assert_failure # category does not exist

  run govc tags.detach -c "$category_name" "$tag_name" "$object"
  assert_success # detach using name instead of ID

  run govc tags.rm -c "$category_name" "$tag"
  assert_success

  run govc tags.category.rm "$category"
  assert_success
}

@test "tags.example" {
  vcsim_env -dc 2 -cluster 2

  govc tags.category.create -d "Kubernetes region" k8s-region

  for region in EMEA US ; do
    govc tags.create -d "Kubernetes region $region" -c k8s-region k8s-region-$region
  done

  govc tags.attach k8s-region-EMEA /DC0
  govc tags.attach k8s-region-US /DC1

  govc tags.category.create -d "Kubernetes zone" k8s-zone

  for zone in DE CA WA ; do
    govc tags.create -d "Kubernetes zone $zone" -c k8s-zone k8s-zone-$zone
  done

  govc tags.attach k8s-zone-DE /DC0/host/DC0_C0
  govc tags.attach k8s-zone-DE /DC0/host/DC0_C1

  govc tags.attach k8s-zone-CA /DC1/host/DC1_C0
  govc tags.attach k8s-zone-WA /DC1/host/DC1_C1

  govc tags.category.ls
  govc tags.category.info

  govc tags.ls
  govc tags.ls -c k8s-region
  govc tags.ls -c k8s-zone
  govc tags.info

  govc tags.attached.ls k8s-region-US
  govc tags.attached.ls k8s-zone-CA
  govc tags.attached.ls -r /DC1
  govc tags.attached.ls -r /DC1/host/DC1_C0
}
