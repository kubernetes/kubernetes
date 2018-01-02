# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

@test "Test overlay network with etcd" {
    skip_for_circleci
    test_overlay etcd
}
