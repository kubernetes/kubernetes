# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

@test "Test overlay network hostmode with consul" {
    skip_for_circleci
    test_overlay_hostmode consul
}
