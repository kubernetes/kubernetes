#!/bin/bash

echo "Deleting Authenticator replication controller" && lmktfyctl stop rc authenticator-controller
echo "Deleting Phabricator service" && lmktfyctl delete -f phabricator-service.json
echo "Deleting Phabricator replication controller" && lmktfyctl stop rc phabricator-controller

echo "Delete firewall rule" && gcloud compute firewall-rules delete -q phabricator-node-80

