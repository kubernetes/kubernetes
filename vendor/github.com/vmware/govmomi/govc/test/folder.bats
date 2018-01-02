#!/usr/bin/env bats

load test_helper

@test "folder.info" {
    for name in / vm host network datastore ; do
        run govc folder.info $name
        assert_success

        govc folder.info -json $name
        assert_success
    done

    result=$(govc folder.info '*' | grep Name: | wc -l)
    [ $result -eq 4 ]

    run govc info.info /enoent
    assert_failure
}

@test "folder.create" {
    vcsim_env

    name=$(new_id)

    # relative to $GOVC_DATACENTER
    run govc folder.create $name
    assert_failure

    run govc folder.create vm/$name
    assert_success

    run govc folder.info vm/$name
    assert_success

    run govc folder.info /$GOVC_DATACENTER/vm/$name
    assert_success

    run govc object.destroy vm/$name
    assert_success

    unset GOVC_DATACENTER
    # relative to /

    run govc folder.create $name
    assert_success

    run govc folder.info /$name
    assert_success

    child=$(new_id)
    run govc folder.create $child
    assert_success

    run govc folder.info /$name/$child
    assert_failure

    run govc object.mv $child /$name
    assert_success

    run govc folder.info /$name/$child
    assert_success

    new=$(new_id)
    run govc object.rename /$name $new
    assert_success
    name=$new

    run govc folder.info /$name
    assert_success

    run govc object.destroy $name
    assert_success
}
