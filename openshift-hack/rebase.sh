#!/bin/bash

# READ FIRST BEFORE USING THIS SCRIPT
#
# This script requires jq, git, podman and bash to work properly (dependencies are checked for you).
# The Github CLI "gh" is optional, but convenient to create a pull request automatically at the end.
#
# This script generates a git remote structure described in:
# https://github.com/openshift/kubernetes/blob/master/REBASE.openshift.md#preparing-the-local-repo-clone
# Please check if you have configured the correct remotes, otherwise the script will fail.
#
# The usage is described in /Rebase.openshift.md.

# validate input args --k8s-tag=v1.21.2 --openshift-release=release-4.8 --bugzilla-id=2003027
k8s_tag=""
openshift_release=""
bugzilla_id=""

usage() {
  echo "Available arguments:"
  echo "  --k8s-tag            (required) Example: --k8s-tag=v1.21.2"
  echo "  --openshift-release  (required) Example: --openshift-release=release-4.8"
  echo "  --bugzilla-id        (optional) creates new PR against openshift/kubernetes:${openshift-release}: Example: --bugzilla-id=2003027"
}

for i in "$@"; do
  case $i in
  --k8s-tag=*)
    k8s_tag="${i#*=}"
    shift
    ;;
  --openshift-release=*)
    openshift_release="${i#*=}"
    shift
    ;;
  --bugzilla-id=*)
    bugzilla_id="${i#*=}"
    shift
    ;;
  *)
    usage
    exit 1
    ;;
  esac
done

if [ -z "${k8s_tag}" ]; then
  echo "Required argument missing: --k8s-tag"
  echo ""
  usage
  exit 1
fi

if [ -z "${openshift_release}" ]; then
  echo "Required argument missing: --openshift-release"
  echo ""
  usage
  exit 1
fi

echo "Processed arguments are:"
echo "--k8s_tag=${k8s_tag}"
echo "--openshift_release=${openshift_release}"
echo "--bugzilla_id=${bugzilla_id}"

# prerequisites (check git, podman, ... is present)
if ! command -v git &>/dev/null; then
  echo "git not installed, exiting"
  exit 1
fi

if ! command -v jq &>/dev/null; then
  echo "jq not installed, exiting"
  exit 1
fi

if ! command -v podman &>/dev/null; then
  echo "podman not installed, exiting"
  exit 1
fi

# make sure we're in "kubernetes" dir
if [[ $(basename "$PWD") != "kubernetes" ]]; then
  echo "Not in kubernetes dir, exiting"
  exit 1
fi

origin=$(git remote get-url origin)
if [[ "$origin" =~ .*kubernetes/kubernetes.* || "$origin" =~ .*openshift/kubernetes.* ]]; then
  echo "cannot rebase against k/k or o/k! found: ${origin}, exiting"
  exit 1
fi

# fetch remote https://github.com/kubernetes/kubernetes
git remote add upstream git@github.com:kubernetes/kubernetes.git
git fetch upstream --tags -f
# fetch remote https://github.com/openshift/kubernetes
git remote add openshift git@github.com:openshift/kubernetes.git
git fetch openshift

#git checkout --track "openshift/$openshift_release"
git pull openshift "$openshift_release"

git merge "$k8s_tag"
# shellcheck disable=SC2181
if [ $? -eq 0 ]; then
  echo "No conflicts detected. Automatic merge looks to have succeeded"
else
  # commit conflicts
  git commit -a
  # resolve conflicts
  git status
  # TODO(tjungblu): we follow-up with a more automated approach:
  # - 2/3s of conflicts stem from go.mod/sum, which can be resolved deterministically
  # - the large majority of the remainder are vendor/generation conflicts
  # - only very few cases require manual intervention due to conflicting business logic
  echo "Resolve conflicts manually in another terminal, only then continue"

  # wait for user interaction
  read -n 1 -s -r -p "PRESS ANY KEY TO CONTINUE"

  # TODO(tjungblu): verify that the conflicts have been resolved
  git commit -am "UPSTREAM: <drop>: manually resolve conflicts"
fi

# openshift-hack/images/hyperkube/Dockerfile.rhel still has FROM pointing to old tag
# we need to remove the prefix "v" from the $k8s_tag to stay compatible
sed -i -E "s/(io.openshift.build.versions=\"kubernetes=)(1.[1-9]+.[1-9]+)/\1${k8s_tag:1}/" openshift-hack/images/hyperkube/Dockerfile.rhel
go_mod_go_ver=$(grep -E 'go 1\.[1-9][0-9]?' go.mod | sed -E 's/go (1\.[1-9][0-9]?)/\1/')
tag="rhel-8-release-golang-${go_mod_go_ver}-openshift-${openshift_release#release-}"

# update openshift go.mod dependencies
sed -i -E "/=>/! s/(\tgithub.com\/openshift\/[a-z|-]+) (.*)$/\1 $openshift_release/" go.mod

echo "> go mod tidy && hack/update-vendor.sh"
podman run -it --rm -v "$(pwd):/go/k8s.io/kubernetes:Z" \
  --workdir=/go/k8s.io/kubernetes \
  "registry.ci.openshift.org/openshift/release:$tag" \
  go mod tidy && hack/update-vendor.sh

# shellcheck disable=SC2181
if [ $? -ne 0 ]; then
  echo "updating the vendor folder failed, is any dependency missing?"
  exit 1
fi

podman run -it --rm -v "$(pwd):/go/k8s.io/kubernetes:Z" \
  --workdir=/go/k8s.io/kubernetes \
  "registry.ci.openshift.org/openshift/release:$tag" \
  make update OS_RUN_WITHOUT_DOCKER=yes

git add -A
git commit -m "UPSTREAM: <drop>: hack/update-vendor.sh, make update and update image"

remote_branch="rebase-$k8s_tag"
git push origin "$openshift_release:$remote_branch"

XY=$(echo "$k8s_tag" | sed -E "s/v(1\.[0-9]+)\.[0-9]+/\1/")
ver=$(echo "$k8s_tag" | sed "s/\.//g")
link="https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG/CHANGELOG-$XY.md#$ver"
if [ -n "${bugzilla_id}" ]; then
  if command -v gh &>/dev/null; then
    XY=$(echo "$k8s_tag" | sed -E "s/v(1\.[0-9]+)\.[0-9]+/\1/")
    ver=$(echo "$k8s_tag" | sed "s/\.//g")
    link="https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG/CHANGELOG-$XY.md#$ver"

    # opens a web browser, because we can't properly create PRs against remote repositories with the GH CLI (yet):
    # https://github.com/cli/cli/issues/2691
    gh pr create \
      --title "Bug $bugzilla_id: Rebase $k8s_tag" \
      --body "CHANGELOG $link" \
      --web

  fi
fi
