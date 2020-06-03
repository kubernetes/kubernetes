#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE}")/lib/init.sh"

function cleanup() {
    return_code=$?
    os::test::junit::generate_report
    os::util::describe_return_code "${return_code}"
    exit "${return_code}"
}
trap "cleanup" EXIT

if ! git status &> /dev/null; then
  os::log::fatal "Not a Git repository"
fi

os::util::ensure::built_binary_exists 'commitchecker'

os::test::junit::declare_suite_start "verify/upstream-commits"
os::cmd::expect_success "commitchecker --start ${PULL_BASE_SHA:-master}"
os::test::junit::declare_suite_end
