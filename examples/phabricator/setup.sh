#!/bin/bash

echo "Create Phabricator replication controller" && kubectl create -f phabricator-controller.json
echo "Create Phabricator service" && kubectl create -f phabricator-service.json
echo "Create Authenticator replication controller" && kubectl create -f authenticator-controller.json
echo "Create firewall rule" && gcloud compute firewall-rules create phabricator-node-80 --allow=tcp:80 --target-tags kubernetes-minion

