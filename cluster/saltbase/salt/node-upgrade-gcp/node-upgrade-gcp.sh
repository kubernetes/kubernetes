#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Monitors node configuration data from the metadata server and runs node
# upgrades when they change.

# Settings
# CACHE_DIR is where metadata files are saved once they've been curled.
readonly CACHE_DIR="metadata-cache"
# *_EXT denote the role of a saved metadata file.
readonly CACHE_EXT="cache"
readonly LATEST_EXT="latest"
# WATCH_KEYS are the metadata keys for which, if any of their values change, a
# node upgrade is performed.
readonly WATCH_KEYS=("kube-env" "startup-script")

# log prints to stdout all arguments received, prepending a prefix (such as the
# date) for more readable logs.
#
# Args:
#   $@ -- echos all arguments received
function log() {
  echo "$(date): ${@}"
}

# get-metadata curls all keys in WATCH_KEYS from the metadata server, retrying
# until success, and saves them in CACHE_DIR named by the key and the provided
# extension.
#
# Assumed vars:
#   WATCH_KEYS
#   CACHE_DIR
#
# Args:
#  $1 -- extension to use for files (e.g. CACHE_EXT or LATEST_EXT)
function get-metadata() {
  local ext="${1}"
  log "Attempting to get metadata."
  for k in "${WATCH_KEYS[@]}"; do
    log "Attempting to get metadata with key '${k}'."
    local cache_file="${CACHE_DIR}/${k}.${ext}"
    # We use --retry for some exponential backoff, but this curls until success.
    until curl -f -s --retry 5 -o "${cache_file}" -H "Metadata-Flavor: Google" \
      "http://metadata/computeMetadata/v1/instance/attributes/${k}"; do :; done
    log "Successfully saved metadata with key '${k}' in '${cache_file}'."
  done
}

# monitor-upgradability runs until any metadata on the server changes, or until
# the request to the metadata server fails. If the request succeeds and we get
# a metadata change, it will kick off a node upgrade if any of the values
# retrieved by a key in WATCH_KEYS changes. This process exits if a node upgrade
# is kicked off. Note that for any case other than performing an upgrade, it
# returns, and thus must be called repeatedly to continue checking for upgrades.
#
# Assumed vars:
#   WATCH_KEYS
#   CACHE_DIR
#   CACHE_EXT
#   LATEST_EXT
function monitor-upgradability() {
  # Wait for _any_ metadata to change.
  log "Waiting for metadata changes."
  curl -s "http://metadata/computeMetadata/v1/instance/attributes/?recursive=true&wait_for_change=true" -H "Metadata-Flavor: Google"
  result=$?
  if [[ ${result} -ne 0 ]]; then
    log "Metadata request failed; returned ${result}"
    return
  fi

  # Get latest metadata.
  log "Metadata changed; getting latest versions."
  get-metadata "${LATEST_EXT}"

  # Diff cache vs latest.
  log "Diffing cached versus latest metadata."
  local do_upgrade=0
  for k in "${WATCH_KEYS[@]}"; do
    diff "${CACHE_DIR}/${k}.${CACHE_EXT}" "${CACHE_DIR}/${k}.${LATEST_EXT}" > /dev/null
    if [[ $? -ne 0 ]]; then
      log "Metadata with key '${k}' differ."
      do_upgrade=1
      break
    fi
  done

  # Don't upgrade if no metadata changed.
  if [[ ${do_upgrade} -eq 0 ]]; then
    log "No metadata differences found; will not upgrade."
    return
  fi

  # Kick off a node upgrade.
  log "Relevant metadata differ; performing node upgrade."
  upgrade-node
}

# upgrade-node launches a node upgrade, assuming that the script to do so is
# found in ${CACHE_DIR}/startup-script.${LATEST_EXT}, then exits.
#
# Assumed vars:
#   CACHE_DIR
#   LATEST_EXT
function upgrade-node() {
  log "Running ${CACHE_DIR}/startup-script.${LATEST_EXT}"
  # Wait 1 second (generous) for this upgrader to exit.
  sudo /bin/bash -c "sleep 1; /bin/bash '${CACHE_DIR}/startup-script.${LATEST_EXT}' --push" &
  # The upgrade must start this process again.
  log "Exiting"
  exit 0
}

# Execution begins here.

# This is running in a root-owned directory, so we do two sudo calls to create
# a nonprivileged one.
log "Creating cache directory '${CACHE_DIR}/'"
sudo mkdir -p "${CACHE_DIR}"
sudo chown $(whoami) "${CACHE_DIR}"

# If we just did an upgrade, use the upgraded-to values as the current cache.
# Otherwise, grab the current metadata as the cache (before the first upgrade,
# any metadata changes that happens before this runs won't be caught).
log "Checking for cached 'latest' configuration metadata."
latest_exists=1
for k in "${WATCH_KEYS[@]}"; do
  if [[ ! -s "${CACHE_DIR}/${k}.${LATEST_EXT}" ]]; then
    latest_exists=0
  fi
done
if [[ latest_exists -eq 1 ]]; then
  log "Found 'latest' versions of all required metadata. Using as current config."
  for k in "${WATCH_KEYS[@]}"; do
    mv "${CACHE_DIR}/${k}.${LATEST_EXT}" "${CACHE_DIR}/${k}.${CACHE_EXT}"
  done
else
  log "One or more 'latest' metadata files not found locally. Getting from server."
  get-metadata "${CACHE_EXT}"
fi

# Wait indefinitely for upgrades.
while true; do
  monitor-upgradability
done
