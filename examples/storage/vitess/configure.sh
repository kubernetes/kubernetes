#!/bin/bash

# Copyright 2017 Google Inc.
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

# This script generates config.sh, which is a site-local config file that is not
# checked into source control.

# Erase any existing file since we append from here on.
> config.sh

# Custom Docker image.
read -p "Vitess Docker image (leave empty for default) []: "
echo "vitess_image=\"$REPLY\"" >> config.sh

# Select and configure Backup Storage Implementation.
storage=gcs
read -p "Backup Storage (file, gcs) [gcs]: "
if [ -n "$REPLY" ]; then storage="$REPLY"; fi

case "$storage" in
gcs)
  # Google Cloud Storage
  read -p "Google Cloud Storage bucket for Vitess backups: " bucket
  if [ -z "$bucket" ]; then
    echo "ERROR: Bucket name must not be empty."
    exit 1
  fi
  echo
  echo "NOTE: If you haven't already created this bucket, you can do so by running:"
  echo "      gsutil mb gs://$bucket"
  echo

  backup_flags=$(echo -backup_storage_implementation gcs \
                      -gcs_backup_storage_bucket "'$bucket'")
  ;;
file)
  # Mounted volume (e.g. NFS)
  read -p "Root directory for backups (usually an NFS mount): " file_root
  if [ -z "$file_root" ]; then
    echo "ERROR: Root directory must not be empty."
    exit 1
  fi
  echo
  echo "NOTE: You must add your NFS mount to the vtctld-controller-template"
  echo "      and vttablet-pod-template as described in the Kubernetes docs:"
  echo "      http://kubernetes.io/v1.0/docs/user-guide/volumes.html#nfs"
  echo

  backup_flags=$(echo -backup_storage_implementation file \
                      -file_backup_storage_root "'$file_root'")
  ;;
*)
  echo "ERROR: Unsupported backup storage implementation: $storage"
  exit 1
esac
echo "backup_flags=\"$backup_flags\"" >> config.sh

