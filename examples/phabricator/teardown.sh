#!/bin/bash

echo "Deleting Authenticator replication controller" && kubectl stop rc authenticator-controller
echo "Deleting Phabricator service" && kubectl delete -f phabricator-service.json
echo "Deleting Phabricator replication controller" && kubectl stop rc phabricator-controller

echo "Delete firewall rule" && gcloud compute firewall-rules delete -q phabricator-node-80

