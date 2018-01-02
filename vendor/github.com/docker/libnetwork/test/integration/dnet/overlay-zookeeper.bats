# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

@test "Test overlay network with zookeeper" {
    skip_for_circleci
    test_overlay zookeeper
}
