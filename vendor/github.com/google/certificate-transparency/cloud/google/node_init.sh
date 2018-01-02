#! /bin/bash
export DATA_DIR=/data
export DATA_DEV=/dev/disk/by-id/google-persistent-disk-1

sudo mkdir ${DATA_DIR}
sudo /usr/share/google/safe_format_and_mount \
  -m "mkfs.ext4 -F" ${DATA_DEV} ${DATA_DIR}

# Install google-fluentd which pushes application log files up into the Google
# Cloud Logs Monitor.
EXPECTED_SHA256="48a50746b23c1deb7ec4b5e377631e2c77b95ecada5eda2bc8986a1e04359141  ./google-fluentd-install.sh"
curl -sSO https://storage.googleapis.com/signals-agents/logging/google-fluentd-install.sh
if ! echo "${EXPECTED_SHA256}" | sha256sum --quiet -c; then
  echo "Got google-fluentd-install.sh with sha256sum "
  sha256sum ./google-fluentd-install.sh
  echo "But expected:"
  echo "${EXPECTED_SHA256}"
  echo "google-fluentd-install.sh may have been updated, verify the new sum at"
  echo "https://cloud.google.com/logging/docs/agent/installation and update"
  echo "this script with the new sha256sum if necessary."
  exit 1
fi

sudo bash ./google-fluentd-install.sh
cat > /tmp/ct-info.conf <<EOF
<source>
  type tail
  format none
  path /data/ctlog/logs/ct-server.*.INFO.*
  pos_file /data/ctlog/logs/ct-server.INFO.pos
  read_from_head true
  tag ct-info
</source>
<source>
  type tail
  format none
  path /data/ctlog/logs/ct-server.*.ERROR.*
  pos_file /data/ctlog/logs/ct-server.ERROR.pos
  read_from_head true
  tag ct-warn
</source>
<source>
  type tail
  format none
  path /data/ctlog/logs/ct-server.*.WARNING.*
  pos_file /data/ctlog/logs/ct-server.WARNING.pos
  read_from_head true
  tag ct-warn
</source>
<source>
  type tail
  format none
  path /data/ctlog/logs/ct-server.*.FATAL.*
  pos_file /data/ctlog/logs/ct-server.FATAL.pos
  read_from_head true
  tag ct-error
</source>
EOF
sudo cp /tmp/ct-info.conf /etc/google-fluentd/config.d/ct-info.conf
sudo service google-fluentd restart
# End google-fluentd stuff
